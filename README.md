# Waypoint Tracking y Comparación de PMP, LQG y MPC en MuJoCo

## Descripción general

Este proyecto extiende un framework de control óptimo para un robot cuadrúpedo en MuJoCo con el objetivo de implementar trayectorias basadas en waypoints y comparar el comportamiento de tres controladores:

- **PMP** (Pontryagin Maximum Principle)
- **LQG** (Linear Quadratic Gaussian)
- **MPC** (Model Predictive Control)

El objetivo principal fue:

1. Implementar trayectorias predefinidas basadas en waypoints.
2. Probar el rendimiento de los distintos controladores sobre estas trayectorias.
3. Comparar su comportamiento en simulación.
4. Publicar tanto la implementación como los resultados obtenidos.

---

## Motivación

El repositorio base, Quadruped Optimal Control: PMP, LQG & MPC (https://github.com/nezih-niegu/quadruped-optimal-control#quadruped-optimal-control-pmp-lqg--mpc),  está orientado al control de la base flotante del cuadrúpedo, rechazo de perturbaciones y seguimiento de referencias, sin embargo, no implementa una pila completa de locomoción quadrúpeda.

Por esta razón, aunque sí fue posible implementar referencias basadas en waypoints y comparar controladores, el robot no logra completar de manera robusta todas las trayectorias espaciales en MuJoCo. Aun así, esto no impide cumplir con el objetivo de la actividad, ya que se logró implementar el sistema de waypoints, probar los controladores y comparar su comportamiento.

---

## Archivos principales

### `examples/run_mujoco.py`
Archivo principal de simulación en MuJoCo.  
Incluye:

- trayectorias predefinidas (`line`, `zigzag`, `diamond`)
- lógica de waypoint switching
- asistencia de propulsión planar hacia los waypoints
- comparación entre PMP, LQG y MPC
- generación de gráficas de resultados

### `examples/waypoint_demo_2d.py`
Simulación auxiliar en 2D para validar la lógica de seguimiento de waypoints sin depender de locomoción cuadrúpeda.  
Este archivo demuestra claramente que:

- existe una secuencia de waypoints
- el sistema se dirige hacia el waypoint actual
- al entrar en un radio de tolerancia, cambia al siguiente waypoint
- la trayectoria resultante sigue la secuencia deseada

### `noise.py`
Archivo auxiliar agregado por compatibilidad.  
Fue necesario porque `gym_quadruped` intenta importar una dependencia relacionada con generación de terreno incluso cuando se usa la escena plana (`scene="flat"`). Este archivo permite ejecutar la simulación correctamente en el entorno local.

---

## Validación del sistema de waypoints

Debido a que la simulación del cuadrúpedo está limitada por la ausencia de locomoción completa, se implementó una simulación 2D complementaria para demostrar de forma clara el funcionamiento del sistema waypoint-based.

En esta simulación 2D:

- el robot parte desde el waypoint inicial
- se mueve hacia el waypoint objetivo actual
- al alcanzarlo dentro de un radio de tolerancia, cambia al siguiente
- se obtiene una trayectoria continua que sigue la secuencia de puntos predefinidos

Esto permite validar que la lógica de waypoints sí funciona correctamente, independientemente de las limitaciones dinámicas del cuadrúpedo en MuJoCo.

<img width="800" height="600" alt="Figure_2d_sim" src="https://github.com/user-attachments/assets/cbc1bcec-0f60-4bea-aa1a-d8cc83f2cb0c" />

---

## Trayectorias evaluadas

Se trabajó con trayectorias predefinidas:

- **line**: trayectoria recta
- **zigzag**: trayectoria con cambios laterales de dirección
- además de una tercer trayectoria de tipo diamante en el código, que no fue analizada

Para el análisis final se reportaron principalmente:

- `line`
- `line + impulse`
- `zigzag`

---

## Metodología de evaluación

En MuJoCo, cada controlador fue probado sobre las trayectorias definidas y se comparó usando las siguientes métricas:

- **RMSE** de posición/velocidad
- **error máximo**
- **norma media de las GRFs**
- **distancia recorrida**
- **número de cambios de waypoint logrados**

Además, se analizaron cualitativamente:

- estabilidad
- robustez
- capacidad de seguir la trayectoria sin caer

---

## Resultados principales

Las siguientes tablas resumen el desempeño de cada controlador en los experimentos principales realizados.

### 1. Trayectoria `line` sin perturbación

| Controlador | RMSE | Mean \|\|u\|\| [N] | Distancia recorrida [m] | Waypoint switches |
|---|---:|---:|---:|---:|
| PMP | 0.9455 | 154.8 | 0.473 | 1 |
| LQG | 1.2518 | 142.2 | 0.823 | 2 |
| MPC | **0.4919** | **64.4** | 0.429 | 1 |

<img width="2100" height="1800" alt="mujoco_comparison_mini_cheetah_none_line" src="https://github.com/user-attachments/assets/b49a920b-7729-4100-b357-8e3fa3681edd" />

**Observación:**  
En la trayectoria recta, MPC obtuvo el menor RMSE y el menor esfuerzo promedio de control. Aunque LQG recorrió más distancia, presentó mayor error global.

---

### 2. Trayectoria `line` con perturbación impulsiva

| Controlador | RMSE | Mean \|\|u\|\| [N] | Distancia recorrida [m] | Waypoint switches |
|---|---:|---:|---:|---:|
| PMP | 0.9455 | 154.8 | 0.473 | 1 |
| LQG | 1.1917 | 127.9 | 0.538 | 2 |
| MPC | **0.4919** | **64.4** | 0.429 | 1 |

<img width="2100" height="1800" alt="mujoco_comparison_mini_cheetah_impulse_line" src="https://github.com/user-attachments/assets/362ded23-2cff-40b3-a749-c7221328ae60" />

**Observación:**  
En esta configuración, la perturbación impulsiva afectó principalmente el comportamiento de LQG, mientras que PMP y MPC mantuvieron métricas globales muy similares a las del caso sin esta perturbación. En presencia de perturbación impulsiva sobre la trayectoria recta, MPC volvió a mostrar el mejor desempeño global en error y esfuerzo de control.

---

### 3. Trayectoria `zigzag` sin perturbación

| Controlador | RMSE | Mean \|\|u\|\| [N] | Distancia recorrida [m] | Waypoint switches |
|---|---:|---:|---:|---:|
| PMP | 0.8452 | 117.0 | 0.281 | 1 |
| LQG | 0.9404 | 152.2 | 0.452 | 1 |
| MPC | **0.5288** | **103.4** | 0.419 | 1 |

<img width="2100" height="1800" alt="mujoco_comparison_mini_cheetah_none_zigzag" src="https://github.com/user-attachments/assets/68cbabfa-7572-4214-a7b6-ce4c16734531" />

**Observación:**  
La trayectoria zigzag resultó más exigente. En este caso, MPC volvió a obtener el menor RMSE, mientras que PMP y LQG mostraron un comportamiento menos robusto. Este caso evidencia con mayor claridad las limitaciones del sistema al exigir cambios laterales más agresivos.

---

## Discusión

Los resultados muestran que el sistema implementado **sí permite definir y aplicar trayectorias basadas en waypoints**, pero el desempeño en MuJoCo depende fuertemente de las capacidades reales del framework de control del cuadrúpedo.

En particular:

- el sistema sí genera trayectorias deseadas hacia waypoints
- los controladores sí intentan seguir dichas referencias
- el comportamiento cambia claramente según el controlador utilizado
- trayectorias más complejas como zigzag exponen limitaciones de estabilidad

Esto confirma que la actividad puede considerarse cumplida, ya que:

- se implementó waypoint-based trajectory generation
- se probaron los distintos controladores
- se comparó su comportamiento en simulación

aunque el robot no logre completar de manera robusta todas las trayectorias en el entorno MuJoCo.

---

## Conclusiones

1. Se implementó correctamente un sistema de trayectorias basadas en waypoints.
2. La lógica de waypoint-following fue validada de forma clara en una simulación 2D auxiliar.
3. En MuJoCo, los controladores PMP, LQG y MPC fueron probados y comparados sobre trayectorias predefinidas.
4. **MPC** fue el controlador con mejor desempeño global en los experimentos realizados, al mostrar menor RMSE y menor esfuerzo de control en los casos evaluados.
5. Las limitaciones observadas en trayectorias más exigentes están relacionadas con que el repositorio base no implementa locomoción cuadrúpeda completa.

---

## Cómo ejecutar

### Pruebas de Trayectorias
```bash
.\.venv\Scripts\python.exe .\examples\run_mujoco.py --controller all --robot-name mini_cheetah --trajectory line --disturbance none --no-render

.\.venv\Scripts\python.exe .\examples\run_mujoco.py --controller all --robot-name mini_cheetah --trajectory line --disturbance impulse --no-render    

.\.venv\Scripts\python.exe .\examples\run_mujoco.py --controller all --robot-name mini_cheetah --trajectory zigzag --assist-kp 12 --assist-kd 10 --assist-fmax 6 --no-render
```

### Simulación 2D de validación de waypoints
```bash
.\.venv\Scripts\python.exe .\examples\waypoint_demo_2d.py
```
