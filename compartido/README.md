# Módulos compartidos

`MM_DATA_VXR1.py` contiene la física y las clases consolidadas que reutilizan los entornos. `TRAIN_VX.py` incorpora flujos de profesor y estudiante y selecciona la familia de experimento por configuración; no debe atribuirse íntegramente a un único agente.

`vx_place_protocol.py` define el observador común de colocación. `vx_place_initials.py` prepara y audita los estados iniciales; `vx_pose_metrics.py` calcula métricas geométricas; `vx_final_student_contract.py` valida la procedencia del estudiante y sus artefactos. `extract_vx_place_robust.py` implementa la extracción pareada del baseline y R1.1.

El preparador restaura los módulos de VX dentro de `.runtime/VX/` y las utilidades restantes en la raíz de `.runtime/`. Los artefactos aprendidos que requieren los contratos se mantienen privados.
