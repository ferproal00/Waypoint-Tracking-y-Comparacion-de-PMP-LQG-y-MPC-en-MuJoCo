import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class WaypointFollower2D:
    def __init__(
        self,
        waypoints,
        dt=0.05,
        max_speed=0.12,
        kp=1.8,
        switch_radius=0.05,
    ):
        self.waypoints = np.array(waypoints, dtype=float)
        self.dt = dt
        self.max_speed = max_speed
        self.kp = kp
        self.switch_radius = switch_radius

        self.pos = self.waypoints[0].copy()
        self.vel = np.zeros(2)
        self.target_idx = 1
        self.path = [self.pos.copy()]
        self.target_history = [self.current_target().copy()]
        self.reached_all = False

    def current_target(self):
        return self.waypoints[min(self.target_idx, len(self.waypoints) - 1)]

    def step(self):
        if self.reached_all:
            self.path.append(self.pos.copy())
            self.target_history.append(self.current_target().copy())
            return

        target = self.current_target()
        error = target - self.pos
        dist = np.linalg.norm(error)

        if dist < self.switch_radius:
            if self.target_idx < len(self.waypoints) - 1:
                self.target_idx += 1
                target = self.current_target()
                error = target - self.pos
                dist = np.linalg.norm(error)
            else:
                self.reached_all = True
                self.vel[:] = 0.0
                self.path.append(self.pos.copy())
                self.target_history.append(target.copy())
                return

        desired_vel = self.kp * error
        speed = np.linalg.norm(desired_vel)
        if speed > self.max_speed:
            desired_vel = desired_vel / speed * self.max_speed

        self.vel = desired_vel
        self.pos = self.pos + self.vel * self.dt

        self.path.append(self.pos.copy())
        self.target_history.append(target.copy())


def run_simulation(
    waypoints,
    total_time=25.0,
    dt=0.05,
    max_speed=0.12,
    kp=1.8,
    switch_radius=0.05,
):
    robot = WaypointFollower2D(
        waypoints=waypoints,
        dt=dt,
        max_speed=max_speed,
        kp=kp,
        switch_radius=switch_radius,
    )

    steps = int(total_time / dt)
    for _ in range(steps):
        robot.step()
        if robot.reached_all:
            break

    return robot


def plot_static(robot, title="Waypoint-Based 2D Demo"):
    path = np.array(robot.path)
    targets = np.array(robot.target_history)
    wps = robot.waypoints

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wps[:, 0], wps[:, 1], "ko--", label="Waypoints")
    ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="Robot path")
    ax.plot(path[0, 0], path[0, 1], "go", markersize=10, label="Start")
    ax.plot(path[-1, 0], path[-1, 1], "ro", markersize=10, label="End")

    for i, wp in enumerate(wps):
        ax.text(wp[0] + 0.01, wp[1] + 0.01, f"W{i}", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def animate_simulation(robot, title="Waypoint-Based 2D Demo"):
    path = np.array(robot.path)
    targets = np.array(robot.target_history)
    wps = robot.waypoints

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wps[:, 0], wps[:, 1], "ko--", label="Waypoints")

    for i, wp in enumerate(wps):
        ax.text(wp[0] + 0.01, wp[1] + 0.01, f"W{i}", fontsize=10)

    trail_line, = ax.plot([], [], "b-", linewidth=2, label="Robot path")
    robot_point, = ax.plot([], [], "ro", markersize=8, label="Robot")
    target_point, = ax.plot([], [], "ms", markersize=8, label="Current target")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    margin = 0.15
    ax.set_xlim(np.min(wps[:, 0]) - margin, np.max(wps[:, 0]) + margin)
    ax.set_ylim(np.min(wps[:, 1]) - margin, np.max(wps[:, 1]) + margin)

    def init():
        trail_line.set_data([], [])
        robot_point.set_data([], [])
        target_point.set_data([], [])
        return trail_line, robot_point, target_point

    def update(frame):
        trail_line.set_data(path[: frame + 1, 0], path[: frame + 1, 1])
        robot_point.set_data([path[frame, 0]], [path[frame, 1]])
        target_point.set_data([targets[frame, 0]], [targets[frame, 1]])
        return trail_line, robot_point, target_point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(path),
        init_func=init,
        interval=40,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Puedes cambiar estos waypoints
    waypoints = [
        (0.00, 0.00),
        (0.25, 0.08),
        (0.50, -0.08),
        (0.75, 0.08),
        (1.00, 0.00),
    ]

    robot = run_simulation(
        waypoints=waypoints,
        total_time=30.0,
        dt=0.05,
        max_speed=0.12,
        kp=1.8,
        switch_radius=0.05,
    )

    print("Waypoints:")
    for i, wp in enumerate(robot.waypoints):
        print(f"  W{i}: {wp}")

    print(f"\nWaypoints reached: {robot.target_idx} / {len(robot.waypoints) - 1}")
    print(f"Reached all: {robot.reached_all}")
    print(f"Final position: {robot.pos}")

    # Gráfica estática
    plot_static(robot, title="2D Waypoint Tracking (Static Result)")

    # Animación
    animate_simulation(robot, title="2D Waypoint Tracking (Animation)")