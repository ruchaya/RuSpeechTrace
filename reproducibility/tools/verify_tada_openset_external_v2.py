#!/usr/bin/env python3
"""Independently diagnose saved TADA open-set predictions (no model training).

Specify --project-root (read-only source) and --output-dir (separate scratch folder).

AUROC, known acceptance and unknown rejection are evaluated against saved values.
AUPR and FPR95 formulas are diagnostic candidates pending comparison with the
experiment's original evaluation code. OSCR is NOT recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# folds_sha256 identifies the complete fixed-fold definition. This is common
# across a dataset's five folds. split_sha256 may differ for EACH fold.
EXPECTED_FOLDS_HASHES = {
    'ruasd': 'cac3bf00c0e8f77a1a2a37a4b3b005c1ce65ba3dbaf364d3b028dd5dde20171d',
    'mlaad': 'eac341ab00a413ff1360ed4324316ff7822a056c4d02717e62b6c9691a66cde0',
}
EXPECTED_FOLD1_SPLIT_HASHES = {
    'ruasd': 'f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0',
    'mlaad': 'f4b66a0c8cbce8e7629d85838eb371d5f7381a9d2eafb63917f8ccf1fe5fbd66',
}
CORE_KEYS = ('auroc', 'known_accept', 'unknown_reject')
CANDIDATE_KEYS = ('aupr_average_precision', 'aupr_trapezoid', 'fpr95_roc_candidate')
TOLERANCE = 1e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0, atol=TOLERANCE))


def verify(dataset: str, fold: int, root: Path) -> dict:
    folder = root / 'results' / 'raw' / 'tada_w2vbert2_layer4_knn21' / dataset / 'open_set'
    predictions_path = folder / f'fold{fold}_test_predictions.csv'
    metrics_path = folder / f'fold{fold}_metrics.json'
    for path in (predictions_path, metrics_path):
        require(path.is_file(), f'Missing file: {path}')

    df = pd.read_csv(predictions_path)
    meta = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
    require(meta['dataset'] == dataset, f'Dataset mismatch: {dataset}, fold {fold}')
    require(int(meta['fold']) == fold, f'Fold mismatch: {dataset}, fold {fold}')
    require(meta['folds_sha256'] == EXPECTED_FOLDS_HASHES[dataset],
            f'Fixed-fold hash mismatch: {dataset}, fold {fold}')

    split_hash = meta.get('split_sha256')
    require(isinstance(split_hash, str) and len(split_hash) == 64
            and all(c in '0123456789abcdef' for c in split_hash.lower()),
            f'Invalid/missing split_sha256: {dataset}, fold {fold}')
    if fold == 1:
        require(split_hash == EXPECTED_FOLD1_SPLIT_HASHES[dataset],
                f'Fold-1 split hash changed: {dataset}')

    required = ['id', 'true_label', 'prediction', 'known_id_score']
    missing = sorted(set(required) - set(df.columns))
    require(not missing, f'Missing prediction columns: {missing}')
    require(not df[required].isna().any().any(),
            f'Null IDs/labels/scores: {dataset}, fold {fold}')
    require(df['id'].is_unique, f'Duplicate IDs: {dataset}, fold {fold}')

    known = set(meta['known_generators'])
    unknown = set(meta['unknown_generators'])
    require(known and unknown and known.isdisjoint(unknown),
            f'Invalid known/unknown generator lists: {dataset}, fold {fold}')
    # Open-set exports may collapse EVERY withheld generator to a single
    # true_label="unknown"; the JSON retains their original generator names.
    # Do not mistake the collapsed label for a generator name mismatch.
    csv_labels = set(df['true_label'])
    observed_known = csv_labels & known
    nonknown_labels = csv_labels - known
    require(observed_known == known,
            f'Known generators missing from test predictions: '
            f'{dataset}, fold {fold}: {sorted(known - observed_known)}')
    if nonknown_labels == {'unknown'}:
        label_encoding = 'collapsed_unknown'
    elif nonknown_labels and nonknown_labels.issubset(unknown):
        label_encoding = 'original_generator_names'
    else:
        raise ValueError(
            f'Unexpected nonknown true labels: {dataset}, fold {fold}: '
            f'{sorted(nonknown_labels)}; expected collapsed "unknown" '
            f'or names from unknown_generators'
        )

    known_mask = df['true_label'].isin(known).to_numpy(dtype=bool)
    unknown_mask = ~known_mask
    require(int(known_mask.sum()) == int(meta['n_known_test']),
            f'Known count mismatch: {dataset}, fold {fold}')
    require(int(unknown_mask.sum()) == int(meta['n_unknown_test']),
            f'Unknown count mismatch: {dataset}, fold {fold}')

    # Stored known_id_score is negative kNN distance. A larger positive
    # distance (-known_id_score) indicates a more likely unknown generator.
    known_scores = pd.to_numeric(df['known_id_score'], errors='raise').to_numpy(dtype=float)
    require(np.isfinite(known_scores).all(),
            f'Non-finite known_id_score: {dataset}, fold {fold}')
    ood_scores = -known_scores
    y_ood = unknown_mask.astype(int)

    auroc = float(roc_auc_score(y_ood, ood_scores))
    ap = float(average_precision_score(y_ood, ood_scores))
    precision, recall, _ = precision_recall_curve(y_ood, ood_scores)
    pr_auc = float(auc(recall, precision))

    # Candidate convention: known generators are positive; select the first
    # empirical ROC point achieving >=95% known TPR. NOT a claimed definition.
    fpr, tpr, _ = roc_curve(known_mask.astype(int), known_scores,
                            drop_intermediate=False)
    indices = np.flatnonzero(tpr >= 0.95)
    require(indices.size > 0, f'ROC never reached TPR 0.95: {dataset}, fold {fold}')
    fpr95_candidate = float(fpr[int(indices[0])])

    # Threshold determined from known-generator VALIDATION only.
    threshold = float(meta['distance_threshold_known_val_q95'])
    require(np.isfinite(threshold), f'Invalid validation threshold: {dataset}, fold {fold}')
    accepted = ood_scores <= threshold
    known_accept = float(accepted[known_mask].mean())
    unknown_reject = float((~accepted[unknown_mask]).mean())

    saved = meta['metrics']
    metrics = {
        'auroc': (auroc, float(saved['auroc'])),
        'known_accept': (known_accept, float(saved['known_accept'])),
        'unknown_reject': (unknown_reject, float(saved['unknown_reject'])),
        'aupr_average_precision': (ap, float(saved['aupr_ood'])),
        'aupr_trapezoid': (pr_auc, float(saved['aupr_ood'])),
        'fpr95_roc_candidate': (fpr95_candidate, float(saved['fpr95'])),
    }
    checks = {name: close(calculated, expected)
              for name, (calculated, expected) in metrics.items()}

    result = {
        'dataset': dataset,
        'fold': fold,
        'predictions_path': predictions_path.relative_to(root).as_posix(),
        'metrics_path': metrics_path.relative_to(root).as_posix(),
        'folds_sha256': meta['folds_sha256'],
        'split_sha256': split_hash,
        'predictions_sha256': sha256_file(predictions_path),
        'metrics_sha256': sha256_file(metrics_path),
        'n_known': int(known_mask.sum()),
        'n_unknown': int(unknown_mask.sum()),
        'true_label_encoding': label_encoding,
        'unknown_generator_membership_verifiable_from_csv': label_encoding == 'original_generator_names',
        'threshold_known_val_q95': threshold,
        'metric_values': {name: {'recomputed': calc, 'saved': orig}
                          for name, (calc, orig) in metrics.items()},
        'checks': checks,
        'core_status': 'PASS' if all(checks[k] for k in CORE_KEYS) else 'FAIL',
    }
    print(
        f'{dataset.upper():5} fold {fold}: '
        f'AUROC={auroc:.8f} (saved={saved["auroc"]:.8f}), '
        f'AP={ap:.8f}, FPR95 candidate={fpr95_candidate:.8f}; '
        f'CORE {result["core_status"]}', flush=True
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True,
                        help='Read-only public release project directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Separate scratch directory for diagnostic output')
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    source = root / 'results' / 'raw' / 'tada_w2vbert2_layer4_knn21'
    if not root.is_dir() or not source.is_dir():
        parser.error(f'Project root or TADA results not found: {source}')
    if output == root or root in output.parents:
        parser.error('Output must be OUTSIDE the public release project')
    if output == source or source in output.parents:
        parser.error('Output must not be inside experimental source data')
    print('Read-only project:', root, flush=True)
    print('Diagnostic outputs:', output, flush=True)
    results = []
    errors = []
    for dataset in ('ruasd', 'mlaad'):
        print(f'\n=== {dataset.upper()} ===', flush=True)
        for fold in range(1, 6):
            try:
                results.append(verify(dataset, fold, root))
            except Exception as exc:
                message = f'{dataset.upper()} fold {fold}: {type(exc).__name__}: {exc}'
                print('ERROR:', message, flush=True)
                errors.append(message)

    output.mkdir(parents=True, exist_ok=True)
    report = {
        'purpose': 'Independent TADA open-set prediction diagnostics',
        'source_root': str(root),
        'output_directory': str(output),
        'method_note': 'OOD score = -known_id_score; unknown class positive for AUROC/AP',
        'scope_note': 'AUPR and FPR95 formulas are diagnostic candidates. '
                      'OSCR, macro-F1, and confidence intervals are NOT checked here. '
                      'For collapsed unknown labels, withheld-generator membership '
                      'cannot be independently confirmed from this CSV alone.',
        'threshold_note': 'Distance threshold supplied by known-generator validation metadata',
        'tolerance': TOLERANCE,
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': sklearn.__version__,
        'results': results,
        'errors': errors,
    }
    json_path = output / 'tada_openset_diagnostic.json'
    csv_path = output / 'tada_openset_diagnostic.csv'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    flat_rows = []
    for result in results:
        row = {key: value for key, value in result.items()
               if key not in ('metric_values', 'checks')}
        for name, values in result['metric_values'].items():
            row[f'{name}_recomputed'] = values['recomputed']
            row[f'{name}_saved'] = values['saved']
        row.update({f'check_{key}': val for key, val in result['checks'].items()})
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print('\n=== FINAL DIAGNOSTIC ===')
    print('Folds analyzed:', len(results), '/ 10')
    for key in CORE_KEYS + CANDIDATE_KEYS:
        print(f'{key}: {sum(r["checks"][key] for r in results)}/10 matches')
    print('Errors:', len(errors))
    print('JSON:', json_path)
    print('CSV:', csv_path)
    core_ok = len(results) == 10 and not errors and all(
        r['core_status'] == 'PASS' for r in results
    )
    print('CORE DIAGNOSTIC:', 'PASS' if core_ok else 'REVIEW REQUIRED')
    print('AUPR/FPR95: diagnostic candidates; OSCR: not checked')
    return 0 if core_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
