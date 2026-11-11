# Manipulación móvil con aprendizaje por refuerzo

Códigos asociados a la tesis *Manipulación móvil utilizando técnicas de inteligencia artificial*, de Claudio Isaac Cortés Mariqueo, Universidad Tecnológica Metropolitana (2026).

El repositorio contiene los códigos organizados por experimento, sus módulos compartidos, los mapas de navegación y el modelo físico del manipulador móvil en MuJoCo. La comparación principal del escrito corresponde al controlador clásico, R1.1 y R3; los desarrollos R2/R2.1 se identifican aparte.

**Los pesos aprendidos, checkpoints, normalizadores y buffers son privados y no se distribuyen en este repositorio.** El modelo físico XML/STL describe el robot y se distingue de los parámetros aprendidos por los agentes.

## Experimentos

| Carpeta | Experimento |
|---|---|
| [baseline](experimentos/baseline) | Controlador clásico |
| [r1](experimentos/r1) | R1: competencia inicial |
| [r1_1](experimentos/r1_1) | R1.1: residual de torque |
| [r2](experimentos/r2) | R2: referencias de velocidad |
| [r2_1](experimentos/r2_1) | R2.1: ajuste de la rama residual |
| [r2_1_acoplado](experimentos/r2_1_acoplado) | R2.1 acoplado |
| [r3](experimentos/r3) | R3: anticipación geométrica |
| [historico_vx](experimentos/historico_vx) | Desarrollo histórico VX |

## Recursos compartidos

- [compartido/](compartido): física y entrenador consolidado VX, observador de colocación, estados iniciales, métricas y contratos de integridad.
- [recursos/RUTAS_SAC/](recursos/RUTAS_SAC): mapas 1–4, rutas RRT*, manifiesto y vistas PNG.
- [recursos/Zenno_Model/](recursos/Zenno_Model): escenas XML, mallas STL y textura del manipulador móvil.
- [ejemplos/](ejemplos): ejemplo Fetch de Gymnasium, independiente del manipulador de la tesis.

## Preparación del código

```bash
python herramientas/verificar_paquete.py
python herramientas/preparar_entorno.py
```

El preparador crea `.runtime/` con las ubicaciones relativas que esperan los programas originales: `VX/`, `VX_R3/`, `RUTAS_SAC/`, `Zenno_Model/` y módulos comunes. No inicia entrenamientos ni sobrescribe archivos locales cuyo contenido haya cambiado.

Los entrenadores y extractores se desarrollaron en Ubuntu; algunos dependen de `fcntl`. Las herramientas de preparación y comprobación funcionan también en Windows. [docs/EJECUCION.md](docs/EJECUCION.md) documenta los puntos de entrada y la instalación de dependencias.

## Visualizar el modelo físico

```bash
python -m pip install mujoco
python herramientas/verificar_paquete.py --mujoco
python -m mujoco.viewer --mjcf=recursos/Zenno_Model/scene.xml
```

La visualización del robot no requiere pesos SAC. Ejecutar una política entrenada sí requiere los artefactos privados del autor; no se proporciona una política de sustitución ni se desactivan las verificaciones de procedencia.

## Integridad y comentarios

[MANIFEST.json](MANIFEST.json) relaciona los códigos y recursos con su origen, SHA-256 y ubicación de ejecución. Las copias originales se conservan fuera del repositorio. Los cambios editoriales realizados afectan solo comentarios y conservan idéntico el árbol sintáctico de Python.

Los módulos sujetos a contratos SHA-256 mantienen sus bytes originales. Los encabezados históricos de algunas clases describen implementaciones intermedias; [docs/COMENTARIOS_Y_PROCEDENCIA.md](docs/COMENTARIOS_Y_PROCEDENCIA.md) aclara los contratos públicos finales y distingue profesores, actores y modelo físico.
