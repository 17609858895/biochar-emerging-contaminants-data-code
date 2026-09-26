"""Additional grouped partitions; original frozen predictions remain untouched."""
import json
import run_model24_topsis as base
from model24_registry import CONFIGS

OUT = base.ROOT / 'evidence/reviewer_validation_v1'
OUT.mkdir(exist_ok=True)
base.OUT = OUT / 'repeated_fits'
base.OUT.mkdir(exist_ok=True)
SEEDS = [91, 131, 173, 211, 257, 307, 353]
protocol = dict(original_seeds=[11,23,47], additional_seeds=SEEDS,
                tracks=['exact','broad'], outer_folds=5, inner_folds=4,
                configurations=CONFIGS, selection='unchanged inner-only rules',
                scope='Repeated partition sensitivity, not independent datasets',
                new_observations=0)
(OUT/'repeat_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
if __name__ == '__main__':
    for seed in SEEDS:
        base.run(seed,['exact','broad'],6)
