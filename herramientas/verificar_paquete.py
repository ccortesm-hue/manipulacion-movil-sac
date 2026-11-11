"""Validate file integrity, Python syntax, routes and MuJoCo resources."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SUFFIXES = {'.zip','.pkl','.pickle','.pt','.pth','.ckpt','.onnx','.safetensors',
                    '.h5','.hdf5','.msgpack','.bin','.npy','.npz','.tar','.gz','.7z'}


def reject_private_artifact(rel):
    path = Path(rel)
    assert path.suffix.lower() not in PRIVATE_SUFFIXES, f'Artefacto privado excluido: {rel}'
    assert not path.suffix.lower().startswith('.ckpt'), f'Checkpoint privado excluido: {rel}'
    assert not {'models','artefactos','data_vx','.runtime'}.intersection(path.parts), f'Directorio privado excluido: {rel}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mujoco', action='store_true', help='Cargar las escenas físicas con MuJoCo instalado.')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    tracked = subprocess.run(['git','-C',str(ROOT),'ls-files','-z'], capture_output=True, check=True).stdout.decode('utf-8').split('\0')
    for rel in filter(None, tracked):
        reject_private_artifact(rel)
    paths = set()
    forbidden = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    for row in manifest['files']:
        path = ROOT/row['path']
        reject_private_artifact(row['path'])
        assert row['path'] not in paths, f"Archivo duplicado: {row['path']}"
        paths.add(row['path'])
        assert path.is_file() and path.stat().st_size == row['bytes'], row['path']
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
        assert len(raw) < 50*1024**2, f'Archivo grande inesperado: {path}'
        if path.suffix == '.py':
            code = raw.decode('utf-8-sig')
            compile(ast.parse(code), str(path), 'exec')
        if path.suffix in ('.py','.json','.csv','.sh'):
            assert not forbidden.search(raw.decode('utf-8-sig')), f'Revisar credencial en {path}'
    routes = []
    for i in range(1,5):
        path = ROOT/'recursos/RUTAS_SAC'/f'map_{i}.json'
        route = json.loads(path.read_text())['path']
        assert len(route) >= 2 and all(len(point)>=2 and all(math.isfinite(float(v)) for v in point[:2]) for point in route)
        routes.append(path.name)
    model_root = ROOT/'recursos/Zenno_Model'
    for path in sorted(model_root.glob('*.xml')):
        document = ET.parse(path).getroot()
        compiler = document.find('compiler')
        meshdir = compiler.get('meshdir','') if compiler is not None else ''
        for element in document.iter():
            file = element.get('file')
            if file:
                candidates = [path.parent/file, path.parent/meshdir/file]
                assert any(p.is_file() for p in candidates), f'Recurso no encontrado: {path.name}: {file}'
    scenes = []
    if args.mujoco:
        import mujoco
        for name in ('scene.xml','scene_base.xml'):
            model = mujoco.MjModel.from_xml_path(str(model_root/name))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            assert model.nu == 12, (name, model.nu)
            scenes.append({'scene': name, 'actuators': model.nu, 'bodies': model.nbody})
        print('MuJoCo:', mujoco.__version__, scenes)
    print('Integridad, sintaxis Python, mapas y recursos XML: correctos. Sin pesos ni normalizadores en el paquete.')
    print('La validación del paquete no ejecuta entrenamiento ni reproduce las métricas de la tesis.')


if __name__ == '__main__':
    main()
