"""Assemble the original import layout from the experiment folders."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', type=Path, default=ROOT/'.runtime')
    args = parser.parse_args()
    target = args.destino.resolve()
    if target == ROOT or ROOT not in target.parents or target.parts[len(ROOT.parts)] != '.runtime':
        parser.error('El destino debe estar dentro de la carpeta .runtime del repositorio.')
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    files = {}
    for row in manifest['files']:
        source = ROOT/row['path']
        if sha(source) != row['sha256']:
            raise RuntimeError(f"El archivo publicado cambió: {row['path']}")
        for rel in row['runtime']:
            dest = (target/rel).resolve()
            if target not in dest.parents:
                raise RuntimeError(f'Ruta de destino fuera del entorno: {rel}')
            if dest in files and files[dest] != row['sha256']:
                raise RuntimeError(f'Dos módulos distintos requieren la misma ruta: {rel}')
            files[dest] = row['sha256']
            if dest.exists():
                if sha(dest) != row['sha256']:
                    raise RuntimeError(f'Se conserva el archivo local modificado: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    print(f'Entorno preparado en {target}')
    print('Los entrenadores y evaluadores se ejecutan desde esta copia; no se inició ninguna corrida.')


if __name__ == '__main__':
    main()
