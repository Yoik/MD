
import os
import sys
import json
import traceback

# 参数（由外部替换）
INTEGRATION_RADIUS = 1.5
ANGLE_SCALE = 30.0
DIST_SCALE = 2.0
OUTPUT_BASE = r'/home/hongyu/MD/1_partial/param_runs/run_r1p5_a30p0_d2p0/results'
MODEL_OUTPUT = r'/home/hongyu/MD/1_partial/param_runs/run_r1p5_a30p0_d2p0/models'
SUMMARY_FILE = r'param_runs/run_r1p5_a30p0_d2p0/summary.json'

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

