# Manipulación móvil con aprendizaje por refuerzo

Código asociado a la tesis *Manipulación móvil utilizando técnicas de inteligencia artificial*, de Claudio Isaac Cortés Mariqueo, Universidad Tecnológica Metropolitana (2026).

## Alcance de esta versión

Esta primera versión reúne los ocho programas disponibles en los anexos del escrito. Los archivos Python conservan su contenido original; sus huellas SHA-256 se registran en [MANIFEST.json](MANIFEST.json).

El paquete todavía está incompleto para reproducir los experimentos. Faltan módulos compartidos y recursos que se encuentran en la computadora del proyecto original. Las comparaciones de la tesis corresponden al controlador clásico, R1.1 y R3.

## Archivos incluidos

| Carpeta | Programas | Función documentada |
|---|---|---|
| `src/baseline/` | `baselines.py`, `DATA_BASELINES.py` | Controlador clásico y extracción de resultados. |
| `src/r1_1/` | `MM_DATA_VXR1.py`, `TRAIN_VX.py` | Ambiente R1.1 y entrenador histórico. |
| `src/r3/` | `VX_R3.py`, `TRAIN_R3.py`, `EVALUATE_R3.py`, `EXTRACT_BASELINE_R11_R3.py` | Intervención geométrica, entrenamiento, evaluación y comparación conjunta. |

## Dependencias pendientes

Entre los archivos importados que todavía no están disponibles se encuentran:

- `MM_DATA_VXR2.py`.
- `extract_vx_place_robust.py`.
- `vx_place_initials.py` y `vx_place_protocol.py`.
- `evaluate_zenno_vx_r1_1.py` y `vx_final_student_contract.py`.

También faltan el inventario de versiones del entorno original, el modelo físico y sus recursos de MuJoCo, las rutas de `RUTAS_SAC` y los manifiestos de selección. Los checkpoints, normalizadores y registros experimentales se documentarán como artefactos separados al incorporarlos.

Los programas utilizan, entre otras bibliotecas, MuJoCo, PyTorch, Gymnasium y Stable-Baselines3. Esta versión no fija versiones ni ofrece instrucciones de reproducción verificadas; se completarán con el paquete original.

## Estado del repositorio

Repositorio: <https://github.com/ccortesm-hue/manipulacion-movil-sac>.

El repositorio es de acceso público para consultar el código asociado al escrito. El paquete todavía está incompleto para reproducir los experimentos; las dependencias pendientes se identifican arriba y en el manifiesto. La licencia de distribución está pendiente de elección por el autor.
