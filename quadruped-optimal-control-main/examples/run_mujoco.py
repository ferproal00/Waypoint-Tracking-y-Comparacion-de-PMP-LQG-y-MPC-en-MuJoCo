#!/usr/bin/env python3
"""Run PMP / LQG / MPC controllers on a quadruped robot in MuJoCo.

This version adds waypoint-propulsion mode:
- predefined spatial waypoints in the XY plane
- waypoint switching by proximity
- bounded planar propulsion-assist force toward the current waypoint
- controller comparison on the same waypoint path

Important:
This is still NOT a full locomotion stack. The robot is assisted toward
the waypoint path while the optimal controllers stabilize the floating base.
"""

import sys
import os
import argparse
import threading
import select
import math
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv

from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController


# =====================================================================
# Nominal physical parameters
# =====================================================================
ROBOT_MASS = 9.0
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])
ROBOT_HIP_HEIGHT = 0.225
ROBOT_FOOT_OFFSET = np.array([
    [0.19,  0.111, -0.225],   # FL
    [0.19, -0.111, -0.225],   # FR
    [-0.19,  0.111, -0.225],  # RL
    [-0.19, -0.111, -0.225],  # RR
])


# =====================================================================
# Teleop
# =====================================================================
@dataclass
class TeleopState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    step_lin: float = 0.05
    step_ang: float = 0.15
    max_vx: float = 0.20
    max_vy: float = 0.15
    max_wz: float = 0.40
    quit_requested: bool = False

    def clamp(self):
        self.vx = float(np.clip(self.vx, -self.max_vx, self.max_vx))
        self.vy = float(np.clip(self.vy, -self.max_vy, self.max_vy))
        self.wz = float(np.clip(self.wz, -self.max_wz, self.max_wz))

    def zero(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0


def teleop_keyboard_loop(teleop: TeleopState):
    """Cross-platform terminal teleop for velocity references."""
    print("\n[Teleop enabled]")
    print("  Up / Down    : forward/backward reference")
    print("  Left / Right : yaw left/right reference")
    print("  z / c        : lateral left/right reference")
    print("  space        : zero commands")
    print("  q            : quit teleop thread\n")

    if os.name == "nt":
        import time
        import msvcrt

        while not teleop.quit_requested:
            if msvcrt.kbhit():
                ch = msvcrt.getch()

                if ch in (b"\x00", b"\xe0"):
                    ch2 = msvcrt.getch()
                    if ch2 == b"H":       # Up
                        teleop.vx += teleop.step_lin
                    elif ch2 == b"P":     # Down
                        teleop.vx -= teleop.step_lin
                    elif ch2 == b"M":     # Right
                        teleop.wz -= teleop.step_ang
                    elif ch2 == b"K":     # Left
                        teleop.wz += teleop.step_ang
                else:
                    ch = ch.decode(errors="ignore").lower()
                    if ch == "z":
                        teleop.vy += teleop.step_lin
                    elif ch == "c":
                        teleop.vy -= teleop.step_lin
                    elif ch == " ":
                        teleop.zero()
                    elif ch == "q":
                        teleop.quit_requested = True

                teleop.clamp()
                print(
                    f"\rref -> vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, wz={teleop.wz:+.2f}   ",
                    end="",
                    flush=True,
                )

            time.sleep(0.02)

    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while not teleop.quit_requested:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)

                    if ch == "\x1b":
                        seq1 = sys.stdin.read(1)
                        seq2 = sys.stdin.read(1)
                        if seq1 == "[":
                            if seq2 == "A":
                                teleop.vx += teleop.step_lin
                            elif seq2 == "B":
                                teleop.vx -= teleop.step_lin
                            elif seq2 == "C":
                                teleop.wz -= teleop.step_ang
                            elif seq2 == "D":
                                teleop.wz += teleop.step_ang
                    elif ch == "z":
                        teleop.vy += teleop.step_lin
                    elif ch == "c":
                        teleop.vy -= teleop.step_lin
                    elif ch == " ":
                        teleop.zero()
                    elif ch == "q":
                        teleop.quit_requested = True

                    teleop.clamp()
                    print(
                        f"\rref -> vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, wz={teleop.wz:+.2f}   ",
                        end="",
                        flush=True,
                    )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print()


# =====================================================================
# Waypoint propulsion
# =====================================================================
@dataclass
class XYWaypoint:
    x: float
    y: float


class WaypointManager:
    """Spatial waypoint manager with sequential switching."""

    def __init__(self, waypoints, switch_radius=0.10):
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints.")
        self.waypoints = [np.array([w.x, w.y], dtype=float) for w in waypoints]
        self.switch_radius = float(switch_radius)
        self.index = 1

    def current_target(self):
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def update(self, p_xy):
        if self.finished():
            return self.current_target(), False

        target = self.current_target()
        reached = np.linalg.norm(target - p_xy) <= self.switch_radius
        if reached and self.index < len(self.waypoints) - 1:
            self.index += 1
            target = self.current_target()
            return target, True
        return target, False

    def finished(self):
        return self.index >= len(self.waypoints) - 1

    def path_array(self):
        return np.array(self.waypoints)


def make_waypoint_path(name: str):
    """
    Softer user-selectable paths.
    Main demonstration path: zigzag.
    """
    if name == "line":
        return [
            XYWaypoint(0.00, 0.00),
            XYWaypoint(0.30, 0.00),
            XYWaypoint(0.60, 0.00),
            XYWaypoint(0.90, 0.00),
        ]

    if name == "zigzag":
        return [
            XYWaypoint(0.00,  0.00),
            XYWaypoint(0.25,  0.08),
            XYWaypoint(0.50, -0.08),
            XYWaypoint(0.75,  0.08),
            XYWaypoint(1.00,  0.00),
        ]

    if name == "diamond":
        return [
            XYWaypoint(0.00, 0.00),
            XYWaypoint(0.25, 0.10),
            XYWaypoint(0.50, 0.00),
            XYWaypoint(0.25, -0.10),
            XYWaypoint(0.00, 0.00),
        ]

    raise ValueError(f"Unknown waypoint path: {name}")


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_waypoint_reference(
    dyn: QuadrupedDynamics,
    x: np.ndarray,
    target_xy: np.ndarray,
    ref_height: float,
    kp_xy: float = 0.45,
    kp_yaw: float = 0.55,
    vmax_xy: float = 0.10,
    wz_max: float = 0.22,
):
    """
    Build x_ref toward the current spatial waypoint.
    """
    pos_xy = x[0:2]
    yaw = x[8]

    err_xy = target_xy - pos_xy
    desired_heading = math.atan2(err_xy[1], err_xy[0]) if np.linalg.norm(err_xy) > 1e-6 else yaw
    yaw_err = wrap_to_pi(desired_heading - yaw)

    v_ref_xy = kp_xy * err_xy
    v_norm = np.linalg.norm(v_ref_xy)
    if v_norm > vmax_xy:
        v_ref_xy = v_ref_xy * (vmax_xy / max(v_norm, 1e-9))

    wz_ref = float(np.clip(kp_yaw * yaw_err, -wz_max, wz_max))

    x_ref = np.zeros(12)
    x_ref[0:2] = target_xy
    x_ref[2] = ref_height
    x_ref[3:5] = v_ref_xy
    x_ref[5] = 0.0
    x_ref[6:8] = 0.0
    x_ref[8] = desired_heading
    x_ref[9:11] = 0.0
    x_ref[11] = wz_ref

    return x_ref, desired_heading, v_ref_xy, wz_ref


def compute_propulsion_assist(
    x: np.ndarray,
    target_xy: np.ndarray,
    kp_force_xy: float = 18.0,
    kd_force_xy: float = 12.0,
    fmax_xy: float = 8.0,
):
    """
    Small planar external force to physically propel the robot toward the waypoint.
    Applied on the floating base in world x,y.
    """
    pos_xy = x[0:2]
    vel_xy = x[3:5]

    err_xy = target_xy - pos_xy
    f_xy = kp_force_xy * err_xy - kd_force_xy * vel_xy

    f_norm = np.linalg.norm(f_xy)
    if f_norm > fmax_xy:
        f_xy = f_xy * (fmax_xy / max(f_norm, 1e-9))

    return f_xy


# =====================================================================
# Helpers
# =====================================================================
def get_state(env) -> np.ndarray:
    """Extract x = [p(3), v(3), rpy(3), ω_body(3)] from the MuJoCo env."""
    p = env.base_pos.copy()
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz.copy()
    omega = env.base_ang_vel(frame="base")
    return np.concatenate([p, v, rpy, omega])


def grf_to_torques(env, grfs: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Convert ground reaction forces to joint torques via Jacobian transpose."""
    tau = np.zeros(env.mjModel.nu)

    try:
        jacobians = env.feet_jacobians(frame="world")
    except Exception:
        return tau

    for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        if not contact[i]:
            continue

        f_leg = grfs[3 * i: 3 * i + 3]
        J_full = jacobians[leg]
        leg_idx = env.legs_qvel_idx[leg]
        J_leg = J_full[:, leg_idx]
        tau_leg = -J_leg.T @ f_leg

        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg

    return tau


def get_contacts(env) -> np.ndarray:
    """Return (4,) boolean contact mask [FL, FR, RL, RR]."""
    try:
        cs, _ = env.feet_contact_state()
        return np.array([cs.FL, cs.FR, cs.RL, cs.RR], dtype=bool)
    except Exception:
        return np.ones(4, dtype=bool)


def get_feet_world(env):
    """Return (4, 3) foot positions in world frame."""
    try:
        fp = env.feet_pos(frame="world")
        return np.array([fp.FL, fp.FR, fp.RL, fp.RR])
    except Exception:
        return None


# =====================================================================
# Dynamics and controllers
# =====================================================================
def build_dynamics():
    dyn = QuadrupedDynamics(
        mass=ROBOT_MASS,
        inertia=ROBOT_INERTIA,
        dt=0.002,
    )
    dyn.r_feet_body = ROBOT_FOOT_OFFSET.copy()
    return dyn


def build_cost_matrices():
    Q = np.diag([
        120, 120, 500,
        16, 16, 45,
        180, 180, 40,
        2, 2, 6,
    ])
    R = np.eye(12) * 2e-4
    Q_f = Q * 5
    return Q, R, Q_f


def build_controller(name: str, dyn: QuadrupedDynamics, Q, R, Q_f, x_ref):
    A_d, B_d, g_d = dyn.get_linear_system(x_ref)
    A_c, B_c = dyn.continuous_AB(x_ref)

    if name == "pmp":
        ctrl = PontryaginController(
            A=A_c,
            B=B_c,
            Q_s=Q,
            R_u=R,
            Q_f=Q_f,
            g_aff=dyn.gravity_vector() / dyn.dt,
            dt=dyn.dt,
            horizon=500,
        )
        ctrl.solve_discrete_sweep(x_ref.copy(), x_ref)
        print("  [PMP] Hamiltonian-based controller initialized")
        return ctrl

    if name == "lqg":
        ctrl = LQGController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_proc=np.diag([1e-3] * 3 + [1e-2] * 3 + [5e-3] * 3 + [1e-2] * 3),
            R_meas=np.diag([5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3),
        )
        ctrl.set_initial_estimate(x_ref)
        print("  [LQG] Controller initialized")
        return ctrl

    if name == "mpc":
        ctrl = MPCController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_f=Q_f * dyn.dt,
            N=10,
            mu=0.6,
            fz_max=150.0,
        )
        print("  [MPC] Horizon=10, OSQP-based controller initialized")
        return ctrl

    raise ValueError(f"Unknown controller: {name}")


def maybe_update_controller_dynamics(controller_name: str, controller, dyn, x_ref, contact, r_feet):
    """Update time-varying linearisation online when supported."""
    try:
        A_c, B_c = dyn.continuous_AB(x_ref, contact, r_feet)
        A_d, B_d = dyn.discretize(A_c, B_c)
        g_d = dyn.gravity_vector()

        if controller_name == "mpc" and hasattr(controller, "update_dynamics"):
            controller.update_dynamics(A_d, B_d, g_d)
        elif controller_name == "lqg":
            controller.A_d = A_d
            controller.B_d = B_d
            controller.g_d = g_d
    except Exception:
        pass


# =====================================================================
# Plotting
# =====================================================================
def save_single_run_plot(result, controller_name, robot_name, disturbance_type, trajectory_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    log_t = result["time"]
    log_x = result["state"]
    log_ref = result["reference"]
    log_u = result["control"]
    log_dist = result["disturbance"]
    path_xy = result["path_xy"]
    target_xy = result["target_xy"]
    wp_xy = result["waypoints"]

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=False)
    fig.suptitle(
        f"{controller_name.upper()} — {robot_name} — {disturbance_type} — path={trajectory_name}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(path_xy[:, 0], path_xy[:, 1], label="robot path")
    axes[0].plot(target_xy[:, 0], target_xy[:, 1], "--", label="active target history")
    axes[0].plot(wp_xy[:, 0], wp_xy[:, 1], "o-", label="waypoints")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    labels_p = [r"$p_x$", r"$p_y$", r"$p_z$"]
    for i in range(3):
        axes[1].plot(log_t, log_x[:, i], label=labels_p[i])
        axes[1].plot(log_t, log_ref[:, i], "--", lw=1.0, label=f"{labels_p[i]} ref")
    axes[1].set_ylabel("Position [m]")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]
    for i in range(3):
        axes[2].plot(log_t, log_x[:, 3 + i], label=labels_v[i])
        axes[2].plot(log_t, log_ref[:, 3 + i], "--", lw=1.0, label=f"{labels_v[i]} ref")
    axes[2].set_ylabel("Velocity [m/s]")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    labels_o = ["roll", "pitch", "yaw"]
    for i in range(3):
        axes[3].plot(log_t, np.degrees(log_x[:, 6 + i]), label=labels_o[i])
        axes[3].plot(log_t, np.degrees(log_ref[:, 6 + i]), "--", lw=1.0, label=f"{labels_o[i]} ref")
    axes[3].set_ylabel("Orientation [deg]")
    axes[3].legend(ncol=3, fontsize=8)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(log_t, np.linalg.norm(log_u, axis=1), label="||GRFs||")
    axes[4].fill_between(log_t, 0, log_dist * 2, alpha=0.25, label="disturbance+assist")
    axes[4].set_ylabel("Force [N]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"results/mujoco_{controller_name}_{robot_name}_{disturbance_type}_{trajectory_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def save_comparison_plot(results, robot_name, disturbance_type, trajectory_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    colors = {"pmp": "#e74c3c", "lqg": "#2ecc71", "mpc": "#3498db"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        f"Controller Comparison — {robot_name} — {disturbance_type} — path={trajectory_name}",
        fontsize=14,
        fontweight="bold",
    )

    for name, data in results.items():
        t = data["time"]
        x = data["state"]
        x_ref = data["reference"]
        u = data["control"]
        path_xy = data["path_xy"]

        pos_err = np.linalg.norm(x[:, :3] - x_ref[:, :3], axis=1)
        vel_err = np.linalg.norm(x[:, 3:6] - x_ref[:, 3:6], axis=1)
        u_norm = np.linalg.norm(u, axis=1)

        axes[0].plot(path_xy[:, 0], path_xy[:, 1], color=colors[name], label=name.upper())
        axes[1].plot(t, pos_err, color=colors[name], lw=1.5)
        axes[2].plot(t, vel_err, color=colors[name], lw=1.5)
        axes[3].plot(t, u_norm, color=colors[name], lw=1.2)

    wp_xy = next(iter(results.values()))["waypoints"]
    axes[0].plot(wp_xy[:, 0], wp_xy[:, 1], "ko--", label="waypoints")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Position error [m]")
    axes[2].set_ylabel("Velocity error [m/s]")
    axes[3].set_ylabel("||GRFs|| [N]")
    axes[3].set_xlabel("Time [s]")

    for ax in axes[1:]:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"results/mujoco_comparison_{robot_name}_{disturbance_type}_{trajectory_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved: {path}")


# =====================================================================
# Main run
# =====================================================================
def run(
    controller_name: str,
    robot_name: str = "mini_cheetah",
    teleop_enabled: bool = False,
    render: bool = True,
    duration: float = 10.0,
    disturbance_type: str = "none",
    save_log: bool = True,
    trajectory_name: str = "zigzag",
    switch_radius: float = 0.10,
    assist_kp: float = 18.0,
    assist_kd: float = 12.0,
    assist_fmax: float = 8.0,
):
    print(f"\n{'=' * 60}")
    print(f"  Controller:   {controller_name.upper()}")
    print(f"  Robot:        {robot_name}")
    print(f"  Teleop:       {teleop_enabled}")
    print(f"  Duration:     {duration}s")
    print(f"  Disturbance:  {disturbance_type}")
    print(f"  Path:         {trajectory_name}")
    print(f"{'=' * 60}\n")

    state_obs_names = tuple(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=state_obs_names,
    )

    _ = env.reset(random=False)
    if render:
        env.render()

    initial_state = get_state(env)
    initial_height = float(initial_state[2])

    teleop = TeleopState()
    if teleop_enabled:
        threading.Thread(
            target=teleop_keyboard_loop,
            args=(teleop,),
            daemon=True,
        ).start()

    dyn = build_dynamics()
    Q, R, Q_f = build_cost_matrices()

    waypoint_manager = None
    if not teleop_enabled:
        raw_waypoints = make_waypoint_path(trajectory_name)
        start_xy = initial_state[0:2].copy()
        first_wp = np.array([raw_waypoints[0].x, raw_waypoints[0].y], dtype=float)
        shifted_waypoints = []
        for wp in raw_waypoints:
            shifted_xy = start_xy + (np.array([wp.x, wp.y], dtype=float) - first_wp)
            shifted_waypoints.append(XYWaypoint(float(shifted_xy[0]), float(shifted_xy[1])))
        waypoint_manager = WaypointManager(shifted_waypoints, switch_radius=switch_radius)
        active_target = waypoint_manager.current_target()
    else:
        active_target = initial_state[0:2].copy()

    x_ref, _, _, _ = build_waypoint_reference(
        dyn=dyn,
        x=initial_state,
        target_xy=active_target,
        ref_height=initial_height,
        kp_xy=0.45,
        kp_yaw=0.55,
        vmax_xy=0.08 if trajectory_name == "zigzag" else 0.10,
        wz_max=0.18 if trajectory_name == "zigzag" else 0.22,
    )

    u_ref = dyn.standing_control()
    controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref)

    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    n_steps = int(duration / sim_dt)

    log_t, log_x, log_x_ref, log_u, log_err, log_dist = [], [], [], [], [], []
    log_path_xy, log_target_xy = [], []
    current_grfs = u_ref.copy()
    waypoint_switches = 0

    print(f"  Sim dt: {sim_dt}s, Ctrl rate: {1 / ctrl_dt:.0f} Hz, Total steps: {n_steps}")
    print(f"  Initial height used as reference: {initial_height:.4f} m")
    print(f"  Waypoint switch radius: {switch_radius:.3f} m")
    print(f"  Assist gains: kp={assist_kp:.1f}, kd={assist_kd:.1f}, fmax={assist_fmax:.1f} N")
    print("  Starting simulation...\n")

    try:
        for step in range(n_steps):
            t = step * sim_dt

            x = get_state(env)
            contact = get_contacts(env)
            r_feet = get_feet_world(env)

            if teleop_enabled:
                cmd_vx = teleop.vx
                cmd_vy = teleop.vy
                cmd_wz = teleop.wz

                x_ref = np.zeros(12)
                x_ref[0:3] = np.array([x[0], x[1], initial_height])
                x_ref[3:6] = np.array([cmd_vx, cmd_vy, 0.0])
                x_ref[6:9] = np.array([0.0, 0.0, x[8]])
                x_ref[9:12] = np.array([0.0, 0.0, cmd_wz])

                active_target = x_ref[0:2].copy()
                assist_xy = np.zeros(2)

            else:
                active_target, switched = waypoint_manager.update(x[0:2])
                if switched:
                    waypoint_switches += 1
                    print(f"  Switched to waypoint {waypoint_manager.index}/{len(waypoint_manager.waypoints)-1}: {active_target}")

                x_ref, _, _, _ = build_waypoint_reference(
                    dyn=dyn,
                    x=x,
                    target_xy=active_target,
                    ref_height=initial_height,
                    kp_xy=0.45,
                    kp_yaw=0.55,
                    vmax_xy=0.08 if trajectory_name == "zigzag" else 0.10,
                    wz_max=0.18 if trajectory_name == "zigzag" else 0.22,
                )
                cmd_vx, cmd_vy, cmd_wz = x_ref[3], x_ref[4], x_ref[11]

                assist_xy = compute_propulsion_assist(
                    x=x,
                    target_xy=active_target,
                    kp_force_xy=assist_kp,
                    kd_force_xy=assist_kd,
                    fmax_xy=assist_fmax,
                )

                assist_scale = min(1.0, t / 1.5)
                assist_xy = assist_scale * assist_xy

            dist = np.zeros(6)
            if disturbance_type == "impulse":
                if 2.0 <= t < 2.15:
                    dist = np.array([50.0, 25.0, 0.0, 0.0, 0.0, 5.0])
            elif disturbance_type == "persistent":
                if t >= 2.0:
                    dist = np.array([15.0, 8.0, 0.0, 0.0, 0.0, 2.0])

            applied_wrench = dist.copy()
            applied_wrench[0:2] += assist_xy
            env.mjData.qfrc_applied[:6] = applied_wrench

            try:
                gyro = env.base_ang_vel(frame="base")
                accel_world = env.base_lin_acc(frame="world")
                R_WB = env.base_configuration[0:3, 0:3]
                accel_body = R_WB.T @ (accel_world - np.array([0.0, 0.0, -9.81]))
                ori_ekf.predict(gyro)
                ori_ekf.update_accel(accel_body)
            except Exception:
                pass

            if step % ctrl_steps == 0:
                maybe_update_controller_dynamics(
                    controller_name=controller_name,
                    controller=controller,
                    dyn=dyn,
                    x_ref=x_ref,
                    contact=contact,
                    r_feet=r_feet,
                )

                try:
                    if controller_name == "lqg":
                        y = x + np.random.randn(12) * np.array(
                            [5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3
                        )
                        current_grfs = controller.step(y, x_ref, u_ref)

                    elif controller_name == "mpc":
                        current_grfs = controller.compute_control(
                            x=x,
                            x_ref=x_ref,
                            u_ref=u_ref,
                            contact_mask=contact,
                        )

                    else:
                        current_grfs = controller.compute_control(
                            x=x,
                            x_ref=x_ref,
                            u_ref=u_ref,
                        )

                except Exception as e:
                    if step < 20:
                        print(f"  Controller warning at t={t:.3f}: {e}")
                    current_grfs = u_ref.copy()

                current_grfs = np.clip(current_grfs, -150.0, 150.0)
                for i in range(4):
                    if not contact[i]:
                        current_grfs[3 * i:3 * i + 3] = 0.0

            tau = grf_to_torques(env, current_grfs, contact)

            try:
                _, _, terminated, _, _ = env.step(action=tau)
            except Exception:
                out = env.step(action=tau)
                terminated = False
                if isinstance(out, tuple) and len(out) >= 3:
                    terminated = bool(out[2])

            if render:
                env.render()

            log_t.append(t)
            log_x.append(x.copy())
            log_x_ref.append(x_ref.copy())
            log_u.append(current_grfs.copy())
            log_err.append(np.linalg.norm(x[:6] - x_ref[:6]))
            log_dist.append(np.linalg.norm(applied_wrench))
            log_path_xy.append(x[0:2].copy())
            log_target_xy.append(active_target.copy())

            if step % int(1.0 / sim_dt) == 0:
                pos_err = np.linalg.norm(x[:3] - x_ref[:3])
                vel_err = np.linalg.norm(x[3:6] - x_ref[3:6])
                wp_dist = np.linalg.norm(active_target - x[0:2])
                print(
                    f"  t={t:5.1f}s | pos_err={pos_err:.4f}m | "
                    f"vel_err={vel_err:.4f}m/s | wp_dist={wp_dist:.3f}m | "
                    f"height={x[2]:.3f}m | vx={x[3]:+.3f} | vy={x[4]:+.3f} | "
                    f"wz={x[11]:+.3f} | ref=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f}) | "
                    f"assist=({assist_xy[0]:+.1f},{assist_xy[1]:+.1f})"
                )

            if terminated:
                print(f"  Terminated at t={t:.2f}s")
                print("  Robot fell or simulation became unstable. Stopping run.")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        teleop.quit_requested = True
        env.close()

    log_t = np.array(log_t) if len(log_t) > 0 else np.zeros(1)
    log_x = np.array(log_x) if len(log_x) > 0 else np.zeros((1, 12))
    log_x_ref = np.array(log_x_ref) if len(log_x_ref) > 0 else np.zeros((1, 12))
    log_u = np.array(log_u) if len(log_u) > 0 else np.zeros((1, 12))
    log_err = np.array(log_err) if len(log_err) > 0 else np.zeros(1)
    log_dist = np.array(log_dist) if len(log_dist) > 0 else np.zeros(1)
    log_path_xy = np.array(log_path_xy) if len(log_path_xy) > 0 else np.zeros((1, 2))
    log_target_xy = np.array(log_target_xy) if len(log_target_xy) > 0 else np.zeros((1, 2))

    if teleop_enabled:
        wp_xy = np.array([initial_state[0:2]])
    else:
        wp_xy = waypoint_manager.path_array()

    result = {
        "time": log_t,
        "state": log_x,
        "reference": log_x_ref,
        "control": log_u,
        "error": log_err,
        "disturbance": log_dist,
        "path_xy": log_path_xy,
        "target_xy": log_target_xy,
        "waypoints": wp_xy,
    }

    if save_log and len(log_t) > 1:
        save_single_run_plot(result, controller_name, robot_name, disturbance_type, trajectory_name)

    path_length = 0.0
    if len(log_path_xy) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(log_path_xy, axis=0), axis=1)))

    print(f"\n  --- {controller_name.upper()} Summary ---")
    print(f"  Position/velocity RMSE: {np.sqrt(np.mean(log_err**2)):.4f}")
    print(f"  Max error: {np.max(log_err):.4f}")
    print(f"  Mean GRF norm: {np.mean(np.linalg.norm(log_u, axis=1)):.1f} N")
    print(f"  Path length traveled: {path_length:.3f} m")
    print(f"  Waypoint switches: {waypoint_switches}")

    return result


# =====================================================================
# Comparison mode
# =====================================================================
def run_comparison(
    render: bool,
    duration: float,
    disturbance_type: str,
    robot_name: str,
    trajectory_name: str = "zigzag",
    switch_radius: float = 0.10,
    assist_kp: float = 18.0,
    assist_kd: float = 12.0,
    assist_fmax: float = 8.0,
):
    results = {}
    for name in ["pmp", "lqg", "mpc"]:
        results[name] = run(
            name,
            robot_name=robot_name,
            teleop_enabled=False,
            render=render,
            duration=duration,
            disturbance_type=disturbance_type,
            save_log=False,
            trajectory_name=trajectory_name,
            switch_radius=switch_radius,
            assist_kp=assist_kp,
            assist_kd=assist_kd,
            assist_fmax=assist_fmax,
        )

    save_comparison_plot(results, robot_name, disturbance_type, trajectory_name)

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY ({disturbance_type}, path={trajectory_name})")
    print(f"{'=' * 60}")
    print(f"  {'Controller':<12} {'RMSE':>10} {'Mean ||u||':>12} {'Path [m]':>10}")
    print(f"  {'-' * 52}")
    for name, data in results.items():
        path_xy = data["path_xy"]
        path_length = 0.0
        if len(path_xy) > 1:
            path_length = float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))
        print(
            f"  {name.upper():<12} "
            f"{np.sqrt(np.mean(data['error']**2)):>10.4f} "
            f"{np.mean(np.linalg.norm(data['control'], axis=1)):>12.1f} "
            f"{path_length:>10.3f}"
        )
    print(f"{'=' * 60}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped waypoint propulsion with MuJoCo")
    parser.add_argument("--controller", default="lqg", choices=["pmp", "lqg", "mpc", "all"])
    parser.add_argument(
        "--robot-name",
        type=str,
        default="mini_cheetah",
        help="Robot name, e.g. mini_cheetah, aliengo, go2, hyqreal",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable keyboard modification of base reference velocities",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--disturbance",
        default="none",
        choices=["impulse", "persistent", "none"],
    )
    parser.add_argument(
        "--trajectory",
        default="zigzag",
        choices=["line", "zigzag", "diamond"],
        help="Spatial waypoint path",
    )
    parser.add_argument("--switch-radius", type=float, default=0.10, help="Waypoint switching distance [m]")
    parser.add_argument("--assist-kp", type=float, default=18.0, help="Planar propulsion proportional gain")
    parser.add_argument("--assist-kd", type=float, default=12.0, help="Planar propulsion damping gain")
    parser.add_argument("--assist-fmax", type=float, default=8.0, help="Planar propulsion max force [N]")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless without viewer",
    )
    args = parser.parse_args()

    do_render = not args.no_render

    if args.controller == "all":
        if args.teleop:
            print("Teleop ignored in comparison mode.")
        run_comparison(
            render=do_render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            robot_name=args.robot_name,
            trajectory_name=args.trajectory,
            switch_radius=args.switch_radius,
            assist_kp=args.assist_kp,
            assist_kd=args.assist_kd,
            assist_fmax=args.assist_fmax,
        )
    else:
        run(
            controller_name=args.controller,
            robot_name=args.robot_name,
            teleop_enabled=args.teleop,
            render=do_render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            save_log=True,
            trajectory_name=args.trajectory,
            switch_radius=args.switch_radius,
            assist_kp=args.assist_kp,
            assist_kd=args.assist_kd,
            assist_fmax=args.assist_fmax,
        )