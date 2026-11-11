# Comentarios y procedencia

Se precisaron dos comentarios históricos: la evaluación inicial del actor clonado antes de SAC y la secuencia de descenso del profesor analítico R1. Se sustituyeron referencias temporales por la descripción del paso realizado. Solo cambian comentarios; el árbol sintáctico permanece idéntico a las fuentes entregadas. Ambas huellas se registran en `MANIFEST.json`.

Los módulos cuya identidad se comprueba mediante SHA-256 conservan sus bytes. Reescribir incluso un comentario en ellos invalidaría su contrato. Para interpretar sus encabezados históricos:

- Los archivos consolidados contienen clases intermedias antes de las definiciones públicas finales. Una observación de 78 componentes en una clase base no representa el contrato final de todos los agentes.
- El actor final R1.1 usa 124 observaciones y 7 acciones: cinco incrementos articulares y dos residuales de torque. El profesor P1 de dos acciones participa en el entrenamiento y no es ese actor final.
- R3 usa 130 observaciones y 6 acciones; su último canal adapta la distancia de anticipación de Pure Pursuit. No representa un torque ni un incremento directo de velocidad.
- Congelar parámetros del brazo no lo inmoviliza físicamente: su política sigue recibiendo observaciones de la dinámica acoplada.
- OFF desactiva la intervención residual del actor correspondiente; no lo convierte en el controlador clásico.
- El trabajo mecánico registrado no mide directamente el consumo eléctrico de batería.

Las copias repetidas recibidas se relacionan mediante `source_aliases`. Las rutas `runtime` permiten preparar la estructura original sin mantener implementaciones distintas no identificadas.
