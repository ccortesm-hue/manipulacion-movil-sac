"""Read-only watcher of one training run; writes a descriptive report, not new evals."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

METHODS = ('baseline', 'reference', 'off', 'on')
LABELS = ('Baseline', 'Inicial', 'Actual OFF', 'Actual ON')
METRICS = {
    'total_time_s': ('Tiempo hasta colocar [s]', 1),
    'total_energy_J': ('Trabajo mecánico absoluto [J]', 1),
    'cte_rms_m': ('CTE RMS [mm]', 1000),
    'yaw_rms_rad': ('Orientación RMS [rad]', 1),
    'cube_goal_dist_xy_m': ('Colocación XY [mm]', 1000),
    'carry_rms_m': ('Distancia cubo–setpoint RMS [mm]', 1000),
    'base_terminal_error_m': ('Posición terminal base [mm]', 1000),
}


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def actor_running(pid, run):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return str(run).encode() in args and any(a.endswith(b'/TRAIN_R21.py') for a in args)


def table(lines, report, title):
    a = report['assessment']
    rows = report['rows']
    lines += ['', f'## {title}', '', '| Métrica | ' + ' | '.join(LABELS) + ' |',
              '|---|---:|---:|---:|---:|']
    rates = []
    for m in METHODS:
        rr = [r for r in rows if r['method'] == m]
        n = sum(bool(r['success_place']) for r in rr)
        rates.append(f'{100*n/len(rr):.1f}% ({n}/{len(rr)})' if rr else 'N/D')
    lines.append('| Éxito, todos los casos | ' + ' | '.join(rates) + ' |')
    for key, (label, scale) in METRICS.items():
        values = a['raw_common_metrics'].get(key, {})
        fmt = lambda x: 'N/D' if x is None else f'{x*scale:.4f}'
        lines.append('| ' + label + ' | ' + ' | '.join(fmt(values.get(m)) for m in METHODS) + ' |')
    lines += ['', f'Las medias corresponden a {a["common_successes"]} éxitos comunes a las cuatro '
              'condiciones. No se mezclan esas medias con la tasa de éxito calculada sobre todos los intentos.',
              '', '| Cambio ON respecto a | Métrica | Variación |', '|---|---|---:|']
    for m, label in (('baseline', 'Baseline'), ('reference', 'Inicial'), ('off', 'OFF')):
        for key, (name, _) in METRICS.items():
            v = a['raw_common_metrics'].get(key, {})
            if v.get(m) is not None and v.get('on') is not None and v[m] > 0:
                lines.append(f'| {label} | {name} | {100*(v["on"]/v[m]-1):+.2f}% |')
    lines += ['', 'Una variación negativa indica una disminución de esa magnitud, no una '
              'mejora global por sí sola. Debe ponderarse junto con éxito y las otras métricas.',
              '', '| Mapa | Baseline | Inicial | OFF | ON |', '|---|---:|---:|---:|---:|']
    for mid in sorted({r['map_id'] for r in rows}):
        entries = []
        for m in METHODS:
            rr = [r for r in rows if r['method'] == m and r['map_id'] == mid]
            entries.append(f'{sum(bool(r["success_place"]) for r in rr)}/{len(rr)}')
        lines.append(f'| {mid} | ' + ' | '.join(entries) + ' |')
    lines += ['', '### Fallos de la política activa', '', '| Mapa | Masa [kg] | Semilla | Motivo registrado |',
              '|---|---:|---:|---|']
    failed = [r for r in rows if r['method'] == 'on' and not r['success_place']]
    for r in failed:
        lines.append(f'| {r["map_id"]} | {r["mass_kg"]} | {r["case_seed"]} | {r["termination_reason"]} |')
    if not failed:
        lines.append('| — | — | — | No se registraron fallos |')
    lines += ['', '### Criterios de selección', '']
    for key in ('qualified', 'residual_physically_active', 'no_success_regression', 'metric_guard',
                'improvement', 'reference_no_success_regression', 'reference_metric_guard', 'reference_improvement'):
        lines.append(f'- `{key}`: `{a.get(key, "N/D")}`.')


def build_report(run, outcome):
    result = read(run/'result.json')
    status = read(run/'status.json') or {}
    reports = []
    for p in sorted(run.glob('block_*/development/summary.json')):
        d = read(p)
        if d and d.get('complete'):
            reports.append((int(p.parts[-3].split('_')[1]), p, d))
    lines = ['# Análisis automático de R2.1 acoplado', '',
             f'Corrida: `{run.name}`.', f'Generado: {datetime.now(timezone.utc).isoformat()}.',
             f'Detección: `{outcome}`. Último estado: `{status.get("stage", "N/D")}`.', '',
             'Informe descriptivo generado a partir de evaluaciones existentes; no ejecuta nuevos episodios '
             'ni modifica el entrenamiento. No demuestra causalidad ni significancia estadística.']
    if not result or not result.get('complete'):
        lines += ['', '**Corrida no confirmada como completa. Los datos siguientes son parciales.**']
    elif result.get('validated_active_residual'):
        lines += ['', '**El candidato cumplió los filtros predefinidos en la evaluación independiente.** '
                  'Eso no implica superioridad en todas las métricas frente al baseline ni autorización '
                  'para sustituir el modelo de la tesis.']
    elif result.get('selected'):
        lines += ['', '**Se seleccionó un candidato en desarrollo, pero no quedó validado por el criterio '
                  'de la evaluación independiente.**']
    else:
        lines += ['', '**Ningún checkpoint cumplió los criterios de selección; no hay residual validado.** '
                  'No confundir la actividad del residual con una mejora de desempeño.']
    lines += ['', '## Evolución en desarrollo', '',
              '| Pasos | Éxito ON | Éxito inicial | Éxito OFF | Éxito baseline | Aceptado |',
              '|---:|---:|---:|---:|---:|---|']
    for steps, _, d in reports:
        a = d['assessment']; rates = a['success_rates']
        lines.append(f'| {steps} | ' + ' | '.join(f'{100*rates[m]:.1f}%' for m in ('on','reference','off','baseline'))
                     + f' | {a["qualified"]} |')
    if reports:
        _, path, d = reports[-1]
        table(lines, d, 'Último checkpoint evaluado — desarrollo')
        lines += ['', f'Fuente: `{path.relative_to(run)}`.']
    if result and result.get('selected'):
        selected = Path(result['selected'])
        selected_summary = selected.parent/'development/summary.json'
        d = read(selected_summary)
        if d and (not reports or selected_summary != reports[-1][1]):
            table(lines, d, 'Checkpoint seleccionado — desarrollo')
        lines += ['', f'Checkpoint seleccionado: `{selected}`.']
    holdout = read(run/'holdout/summary.json')
    if holdout and holdout.get('complete'):
        table(lines, holdout, 'Evaluación independiente — no usada para seleccionar')
    else:
        lines += ['', '**No hay una evaluación independiente completa disponible.** '
                  'Los resultados de desarrollo no deben presentarse como una validación independiente.']
    lines += ['', '## Interpretación y límites', '',
              '- ON/OFF compara el mismo checkpoint, anulando solo la acción de base en OFF. '
              'El movimiento posterior del brazo también puede cambiar por realimentación.',
              '- La comparación con el inicial verifica que una ventaja frente a OFF no proceda '
              'solo de haber deteriorado el controlador sin residual.',
              '- El tiempo y el trabajo se cortan en la colocación estable, sin exigir home. '
              'El trabajo incluye gripper; no es una medición eléctrica.',
              '- El RMS cubo–setpoint mide desviación de la postura de transporte: no prueba '
              'por sí solo caída u oscilaciones.',
              '- Se emparejan estados iniciales, mapas, masas y perturbaciones. Los límites de '
              'aborto no son idénticos: agente 10/45/10 s; baseline 200 s globales.',
              '- Una ventaja temporal que ya estaba en el modelo inicial no puede atribuirse '
              'íntegramente al residual nuevo. No se recomienda continuar o promover un modelo '
              'solo por su porcentaje de éxito.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run/'config.json').is_file():
        parser.error('Run config missing')
    if not args.once and not args.pid:
        parser.error('--pid required for watcher')
    print('WATCH_START', run, flush=True)
    deadline = time.monotonic()+48*3600
    outcome = 'manual_snapshot'
    if not args.once:
        while True:
            result = read(run/'result.json')
            status = read(run/'status.json') or {}
            if result and result.get('complete'):
                outcome = 'finished'; break
            if status.get('stage') in ('ERROR','INTERRUPTED'):
                outcome = status['stage'].lower(); break
            if not actor_running(args.pid,run):
                outcome = 'process_ended_without_completion'; break
            if time.monotonic() >= deadline:
                outcome = 'watch_timeout_not_completed'; break
            time.sleep(120)
    output = args.output or run/'report/analysis_auto.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        f.write(build_report(run,outcome))
    print('WATCH_DONE', outcome, output, flush=True)


if __name__ == '__main__':
    main()
