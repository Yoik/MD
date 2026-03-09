#!/usr/bin/env python3
"""
run_param_sweep.py

并行化参数扫描：测试不同 INTEGRATION_RADIUS、角度衰减尺度、距离衰减尺度
对每组参数运行分析 (run_analysis_v2) -> 生成 results -> 用 train_efficacy_model 训练并收集指标

输出：
 - 每组参数的输出文件夹： ./param_runs/run_r{r}_a{a}_d{d}/results
 - 每组模型输出： ./param_runs/run_r{r}_a{a}_d{d}/models
 - 汇总表： ./param_runs/summary_metrics.csv

用法：
    python run_param_sweep.py
或者自定义网格：
    python run_param_sweep.py --radii 1.0 1.5 --angles 20 30 --dists 1.0 2.0 --workers 4

"""
import os
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


RUNS_DIR = Path('./param_runs')
RUNS_DIR.mkdir(exist_ok=True)


RUNNER_TEMPLATE = """
import os
import sys
import json
import traceback

# 参数（由外部替换）
INTEGRATION_RADIUS = __RADIUS__
ANGLE_SCALE = __ANGLE__
DIST_SCALE = __DIST__
OUTPUT_BASE = r'__OUTPUT_BASE__'
MODEL_OUTPUT = r'__MODEL_OUTPUT__'
SUMMARY_FILE = r'__SUMMARY_FILE__'

try:
    # Add parent directory to path so modules can be imported
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 1) Monkeypatch geometry functions to use provided scales
    import modules.geometry as geom
    import numpy as np

    def calculate_carbon_angles_and_decay(carbon_positions, phe_center, phe_normal):
        angles = np.zeros(6)
        decay_factors = np.ones(6)
        if phe_normal is None:
            return angles, decay_factors
        for i, c_pos in enumerate(carbon_positions):
            vec_to_c = c_pos - phe_center
            norm = np.linalg.norm(vec_to_c)
            if norm < 1e-10:
                angles[i] = 0
                decay_factors[i] = 0
                continue
            dot_prod = np.dot(vec_to_c, phe_normal)
            cos_angle = dot_prod / norm
            angle_rad = np.arccos(np.clip(np.abs(cos_angle), -1, 1))
            angles[i] = np.degrees(angle_rad)
            angle_deviation = abs(angles[i] - 90.0)
            decay_factors[i] = np.exp(-((angle_deviation / ANGLE_SCALE) ** 2))
        return angles, decay_factors

    def calculate_distance_decay(carbon_positions, phe_center, phe_normal):
        import numpy as _np
        if phe_normal is None:
            return _np.zeros(6), _np.ones(6)
        vec_to_atoms = carbon_positions - phe_center
        perp_distances = np.abs(np.dot(vec_to_atoms, phe_normal))
        decay_factors = np.exp(-((perp_distances / DIST_SCALE) ** 2))
        return perp_distances, decay_factors

    geom.calculate_carbon_angles_and_decay = calculate_carbon_angles_and_decay
    geom.calculate_distance_decay = calculate_distance_decay

    # 2) Run analysis (with overridden INTEGRATION_RADIUS and OUTPUT_BASE_DIR)
    import run_analysis_v2 as ra
    ra.INTEGRATION_RADIUS = INTEGRATION_RADIUS
    ra.OUTPUT_BASE_DIR = OUTPUT_BASE
    ra.main()

    # 3) Run training using the generated results
    from train_efficacy_model import EfficacyPredictor
    predictor = EfficacyPredictor(results_dir=os.path.join(OUTPUT_BASE), output_dir=MODEL_OUTPUT)
    predictor.run_pipeline()

    # 4) Collect metrics (model_metrics.csv)
    metrics_file = os.path.join(MODEL_OUTPUT, 'model_metrics.csv')
    summary = {'radius': INTEGRATION_RADIUS, 'angle_scale': ANGLE_SCALE, 'dist_scale': DIST_SCALE, 'metrics_file': metrics_file}
    if os.path.exists(metrics_file):
        import pandas as pd
        df = pd.read_csv(metrics_file, index_col=0)
        # pick best model by LOO_R2 column if exists
        if 'LOO_R2' in df.columns:
            best_row = df['LOO_R2'].astype(float).idxmax()
            summary['best_model'] = best_row
            summary.update(df.loc[best_row].to_dict())
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

except Exception as e:
    err = {'error': str(e), 'trace': traceback.format_exc(), 'radius': INTEGRATION_RADIUS, 'angle_scale': ANGLE_SCALE, 'dist_scale': DIST_SCALE}
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(err, f, indent=2)
    raise

"""


def make_runner(radius, angle_scale, dist_scale, runs_dir: Path):
    tag = f"r{radius}_a{angle_scale}_d{dist_scale}".replace('.', 'p')
    run_dir = runs_dir / f"run_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_base = run_dir / 'results'
    model_output = run_dir / 'models'
    output_base_str = str(output_base.resolve())
    model_output_str = str(model_output.resolve())
    summary_file = run_dir / 'summary.json'

    runner_path = run_dir / 'runner.py'
    content = RUNNER_TEMPLATE
    content = content.replace('__RADIUS__', str(radius))
    content = content.replace('__ANGLE__', str(angle_scale))
    content = content.replace('__DIST__', str(dist_scale))
    content = content.replace('__OUTPUT_BASE__', output_base_str.replace('\\', '\\\\'))
    content = content.replace('__MODEL_OUTPUT__', model_output_str.replace('\\', '\\\\'))
    content = content.replace('__SUMMARY_FILE__', str(summary_file).replace('\\', '\\\\'))
    runner_path.write_text(content)
    return runner_path, run_dir, summary_file


def run_runner(runner_path: Path):
    cmd = ['python', str(runner_path.resolve())]
    # Change to workspace root to ensure module imports work
    cwd = str(runner_path.resolve().parent.parent.parent)  # param_runs/run_x/runner.py -> /home/hongyu/MD/1_partial
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return runner_path.parent, proc.returncode, proc.stdout, proc.stderr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--radii', nargs='+', type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    p.add_argument('--angles', nargs='+', type=float, default=[20.0, 30.0, 40.0])
    p.add_argument('--dists', nargs='+', type=float, default=[1.0, 2.0, 3.0])
    p.add_argument('--workers', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    grid = list(itertools.product(args.radii, args.angles, args.dists))
    print(f"Parameter grid size: {len(grid)}")

    runners = [make_runner(r, a, d, RUNS_DIR) for (r, a, d) in grid]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_runner, runner_path): (runner_path, run_dir, summary_file) for (runner_path, run_dir, summary_file) in runners}
        for fut in as_completed(futures):
            runner_path = futures[fut][0]
            run_dir = futures[fut][1]
            summary_file = futures[fut][2]
            try:
                run_dir_path, code, out, err = fut.result()
                print(f"Completed {run_dir_path.name} (rc={code})")
                if out:
                    print("--- stdout ---")
                    print(out.splitlines()[-5:])
                if err:
                    print("--- stderr (tail) ---")
                    print('\n'.join(err.splitlines()[-10:]))
                # read summary
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    results.append(summary)
            except Exception as e:
                print(f"Runner failed: {e}")

    # 汇总所有结果到CSV
    import pandas as pd
    if results:
        df = pd.json_normalize(results)
        out_csv = RUNS_DIR / 'summary_metrics.csv'
        df.to_csv(out_csv, index=False)
        print(f"Saved summary to {out_csv}")
    else:
        print("No results collected")


if __name__ == '__main__':
    main()
