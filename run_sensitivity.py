import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run_command(args):
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def write_params_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def move_file(src, dst):
    if not src.exists():
        raise FileNotFoundError(f'Missing expected file: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def organize_pigment_run(source_root, output_root, run_id, params):
    pigments = ['indigoidine', 'bikaverin']
    mediums = ['YNB', 'YP']
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_params_json(run_dir / 'params.json', params | {'mediums': mediums, 'kind': 'pigment'})
    for medium in mediums:
        medium_dir = run_dir / medium
        medium_dir.mkdir(parents=True, exist_ok=True)
        for pigment in pigments:
            src = source_root / medium / f'{pigment}_ra_results.csv'
            dst = medium_dir / f'{pigment}_ra_results.csv'
            move_file(src, dst)


def organize_aromatic_run(source_root, output_root, run_id, params, target_aromas, anaerobic):
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_params_json(
        run_dir / 'params.json',
        params | {'target_aromas': target_aromas, 'anaerobic': anaerobic, 'kind': 'aromatic'},
    )
    suffix = 'ana' if anaerobic else 'aer'
    for target in target_aromas:
        target_name = f'{target}_{suffix}'
        src_dir = source_root / target_name
        dst_dir = run_dir / target_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob('*_ra_results.csv'):
            move_file(src, dst_dir / src.name)
        if src_dir.exists():
            try:
                src_dir.rmdir()
            except OSError:
                pass


def run_pigment_sensitivity(job):
    output_root = REPO_ROOT / job.get('output_root', 'results/pigment_sensitivity')
    output_root.mkdir(parents=True, exist_ok=True)
    default_delta = job.get('delta', 1E-2)
    default_epsilon = job.get('epsilon', 1E-4)
    runs = job['runs']
    for idx, run_params in enumerate(runs, start=1):
        params = run_params | {'delta': run_params.get('delta', default_delta), 'epsilon': run_params.get('epsilon', default_epsilon), 'custom_tolerance': 'delta' in run_params or 'epsilon' in run_params}
        run_id = params.get('id', f'ra_{idx:04d}')
        delta = params.get('delta', 1E-2)
        epsilon = params.get('epsilon', 1E-4)
        source_root = output_root / '_cache' / f'delta_{delta:g}_epsilon_{epsilon:g}'
        run_command(
            [
                'run_pigment.py',
                str(params.get('n_mutations', 0)),
                str(params.get('n_samples', 0)),
                str(source_root),
                str(delta),
                str(epsilon),
            ]
        )
        organize_pigment_run(source_root, output_root, run_id, params)


def run_aromatic_sensitivity(job):
    output_root = REPO_ROOT / job.get('output_root', 'results/aromatic_sensitivity')
    output_root.mkdir(parents=True, exist_ok=True)
    default_delta = job.get('delta', 1E-2)
    default_epsilon = job.get('epsilon', 1E-4)
    target_aromas = job['target_aromas']
    if isinstance(target_aromas, str):
        target_aromas = [target.strip() for target in target_aromas.split(',') if target.strip()]
    target_arg = ','.join(target_aromas)
    anaerobic = bool(job.get('anaerobic', False))
    env_input = job.get('env_input', 'data/ale_envs.csv')
    runs = job['runs']
    for idx, run_params in enumerate(runs, start=1):
        params = run_params | {'delta': run_params.get('delta', default_delta), 'epsilon': run_params.get('epsilon', default_epsilon), 'custom_tolerance': 'delta' in run_params or 'epsilon' in run_params}
        run_id = params.get('id', f'ra_{idx:04d}')
        delta = params.get('delta', 1E-2)
        epsilon = params.get('epsilon', 1E-4)
        source_root = output_root / '_cache' / f'delta_{delta:g}_epsilon_{epsilon:g}'
        run_command(
            [
                'run_aromatic.py',
                target_arg,
                'Y' if anaerobic else 'N',
                str(env_input),
                str(params.get('n_mutations', 0)),
                str(params.get('n_samples', 0)),
                str(source_root),
                str(delta),
                str(epsilon),
            ]
        )
        organize_aromatic_run(source_root, output_root, run_id, params, target_aromas, anaerobic)


def main():
    parser = argparse.ArgumentParser(description='Run sensitivity analysis for aromatic or pigment workflows.')
    parser.add_argument('config_file', help='Path to JSON config file.')
    parser.add_argument('kind', choices=['pigment', 'aromatic'], help='Which workflow kind to run from the config file.')
    args = parser.parse_args()

    config = json.loads(Path(args.config_file).read_text(encoding='utf-8'))
    jobs = config.get('jobs', [])
    if not jobs:
        raise ValueError('Config file must contain a non-empty "jobs" list.')

    for job in jobs:
        kind = job.get('kind')
        if kind != args.kind:
            continue
        if kind == 'pigment':
            run_pigment_sensitivity(job)
        elif kind == 'aromatic':
            run_aromatic_sensitivity(job)
        else:
            raise ValueError(f'Unsupported job kind: {kind}')


if __name__ == '__main__':
    main()
