# Ejecución

## Modelo físico

```bash
python -m pip install mujoco
python herramientas/verificar_paquete.py --mujoco
python -m mujoco.viewer --mjcf=recursos/Zenno_Model/scene.xml
```

El verificador carga ambas escenas sin abrir una ventana y comprueba los actuadores. El visor requiere una sesión gráfica y OpenGL. No se necesitan pesos de agentes para visualizar el robot.

## Código experimental en Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python herramientas/preparar_entorno.py
cd .runtime
```

`requirements.txt` enumera las dependencias entregadas, sin atribuirles versiones originales que no se han documentado. La instalación de PyTorch/CUDA depende del equipo. Algunos programas emplean `fcntl` y se ejecutan en Linux. Los lanzadores Bash históricos pueden contener rutas del intérprete original; las interfaces Python permiten configurar las entradas.

| Componente | Punto de entrada dentro de `.runtime` |
|---|---|
| Baseline | `VX/baselines.py`, `VX/DATA_BASELINES.py` |
| R1 | `train_zenno_vx_r1.py`, `evaluate_zenno_vx_r1.py` |
| Profesor P1 de R1.1 | `train_zenno_vx_r1_1.py` |
| Entrenamiento consolidado | `VX/TRAIN_VX.py` |
| R2.1 | `VX_R2_1/TRAIN_R21.py` |
| R2.1 acoplado | `VX_R2_1_COUPLED/TRAIN_R21.py` |
| R3 | `VX_R3/TRAIN_R3.py`, `VX_R3/EVALUATE_R3.py` |
| Comparación de tesis | `VX_R3/analysis/EXTRACT_BASELINE_R11_R3.py` |
| Ablación | `VX_R3/analysis/EXTRACT_RESIDUAL_OFF_R11_R3.py` |

La preparación restaura también `zenno_vx_r2.py` con el contenido de `MM_DATA_VXR2.py`, porque el contrato del estudiante utiliza aquel nombre. No se introduce una política diferente.

## Artefactos privados

El código conserva las referencias a checkpoints, normalizadores y manifiestos de exportación que requiere cada experimento. Estos archivos son privados del autor y no están incluidos. Las rutas históricas absolutas no representan ubicaciones válidas en un nuevo equipo: se deben configurar los artefactos locales al ejecutar los programas.

La comprobación de sintaxis y de recursos no equivale a reproducir los entrenamientos ni las métricas de la tesis. Los scripts de evaluación que necesitan una política no se ejecutan sin sus artefactos privados.

## Ejemplo independiente

`ejemplos/fetch_gymnasium.py` utiliza Gymnasium Robotics y el entorno Fetch. No es la implementación del manipulador móvil de la tesis; su dependencia adicional es `gymnasium-robotics`.
