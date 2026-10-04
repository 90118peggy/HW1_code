"""Read existing experiment evidence and audit confusion matrices without inference."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace/HW1_code')


def read(path):
    content = (ROOT / path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'data': json.loads(content)}


def main():
    evidence = {'collected_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'existing validation artifacts; no model rerun; ALM result records independently audited', 'experiments': {}}
    mapping = {'E0': 'baseline_seed42', 'E1': 'E1_seed42', 'E2': None, 'E3': 'E3_seed42'}
    for dataset in ('A', 'B'):
        for experiment, suffix in mapping.items():
            suffix = suffix or ('E2_layer06' if dataset == 'A' else 'E2_layer05')
            evidence['experiments'][dataset + '_' + experiment] = read(Path('reports') / f'{dataset}_{suffix}_eval/metrics.json')
        evidence['experiments'][dataset + '_E4'] = read(Path('outputs/checkpoints') / f'{dataset}_E4_seed42/reload_validation.json')
        for mode in ('control', 'kd'):
            key = dataset + '_E5_' + mode
            evidence['experiments'][key] = read(Path('reports/E5_final_eval') / f'{dataset}_{mode}/metrics.json')
            evidence['experiments'][key]['predictions_path'] = f'reports/E5_final_eval/{dataset}_{mode}/predictions.csv'
    for name, value in evidence['experiments'].items():
        payload = value['data']
        matrix = payload.get('confusion_matrix')
        if matrix is not None:
            samples = payload.get('samples', payload.get('n_samples'))
            matrix_total = sum(map(sum, matrix))
            trace = sum(matrix[i][i] for i in range(len(matrix)))
            value['matrix_audit'] = {'total': matrix_total, 'top1_correct': trace, 'reconstructed_top1': trace / matrix_total}
            if samples is not None and matrix_total != samples:
                raise ValueError(f'{name}: matrix sample mismatch')
            if 'top1' in payload and abs(trace / matrix_total - payload['top1']) > 1e-6:
                raise ValueError(f'{name}: matrix Top1 mismatch')
    evidence['selected_candidate_configs'] = {}
    for relative in ['outputs/checkpoints/A_E5_control_seed42/run_config.json', 'outputs/checkpoints/B_E2_seed42/config.json', 'outputs/checkpoints/B_E2_seed42/selected_layer.json']:
        evidence['selected_candidate_configs'][relative] = read(relative)
    alm = ROOT / 'reports/ALM_validation_20261004'
    if (alm / 'metrics.json').exists():
        evidence['alm_protocol'] = read('reports/ALM_validation_20261004/protocol.json')
        evidence['alm_metrics'] = read('reports/ALM_validation_20261004/metrics.json')
        for name, group in evidence['alm_metrics']['data']['groups'].items():
            for stage in ('initial', 'final'):
                item = group[stage]
                if sum(map(sum, item['confusion_matrix_with_invalid'])) != item['samples']:
                    raise ValueError(f'ALM {name}/{stage}: matrix count mismatch')
                if sum(row[-1] for row in item['confusion_matrix_with_invalid']) != item['invalid_outputs']:
                    raise ValueError(f'ALM {name}/{stage}: invalid count mismatch')
    evidence['limitations'] = ['Single-seed validation estimates, not held-out test performance.', 'Original source model selection is not repeated or altered by this audit.', 'Pretrained model pretraining-data overlap cannot be excluded.', 'Checkpoint backup location still requires verification; weights were reportedly intentionally removed from this GPU.', 'ALM NF4 quantization applies to language layers; full audio tower remains FP16.', 'ALM prompts are fixed before full scoring, invalid predictions remain in denominator; initial and retry-final metrics reported separately.']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'reports' / f'HW1_evidence_{stamp}.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=False)
    print('EVIDENCE_FILE', destination)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()

