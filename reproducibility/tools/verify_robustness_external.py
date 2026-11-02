"""Read-only consistency verifier for the published RuASD robustness artifacts.

Usage:
  python verify_robustness_external.py --project-root S:\\...\\PUBLIC_RELEASE \
      --output-dir S:\\PUBLIC_RELEASE_SMOKE\\robustness_verification

Recalculates point metrics and reproduces stratified paired bootstrap intervals
from frozen per-recording predictions. Never writes to --project-root.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

MAIN_CONDITIONS = (
    "clean", "noise_30db", "noise_20db", "noise_10db", "synthetic_echo", "mp3_64kbps"
)
MAIN_COMPARISONS = (
    ("noise_30db", "clean"), ("noise_20db", "clean"),
    ("noise_10db", "clean"), ("synthetic_echo", "clean"),
    ("mp3_64kbps", "clean"), ("pcm16_original", "clean"),
    ("pcm16_safe", "clean"), ("mp3_safe", "clean"),
    ("pcm16_safe", "pcm16_original"), ("mp3_safe", "mp3_64kbps"),
    ("mp3_64kbps", "pcm16_original"), ("mp3_safe", "pcm16_safe")
)
SEEDS = (42, 123, 456, 789, 2026)
ATOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(computed, recorded, description):
    require(np.isfinite(float(computed)) and np.isfinite(float(recorded)), f"{description}: non-finite value")
    require(abs(float(computed) - float(recorded)) <= ATOL,
            f"{description}: computed={computed!r} saved={recorded!r}")


def compare_ci(computed, recorded, context):
    if isinstance(recorded, str):
        recorded = ast.literal_eval(recorded)
    require(len(computed) == 2 and len(recorded) == 2, context + ': invalid CI')
    for i, (a, b) in enumerate(zip(computed, recorded)):
        equal_number(a, b, f"{context} bound {i}")


def load_csv(path):
    require(path.is_file(), f"Missing CSV: {path}")
    return pd.read_csv(path)


def load_json(path):
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_selection(root, selected, split_frame):
    require(len(selected) == 680 and selected.file_id.is_unique, "selected TEST must have 680 unique IDs")
    require(selected.generator.nunique() == 34, "selected TEST must have 34 generators")
    require((selected.groupby('generator').size() == 20).all(), "selected TEST must have 20 records per generator")
    frozen = split_frame.set_index('file_id')
    ids = selected.file_id.to_list()
    require(set(ids).issubset(set(frozen.index)), "selected TEST references unknown recording IDs")
    require(frozen.loc[ids, 'split'].eq('test').all(), "selected TEST includes non-test IDs")
    require(np.array_equal(frozen.loc[ids, 'generator'].to_numpy(), selected.generator.to_numpy()), "selected TEST generator mismatch")


def check_predictions(frame, ordered_ids, true, condition, model=None):
    sub = frame.loc[frame.condition.eq(condition)]
    if model is not None:
        sub = sub.loc[sub.model.eq(model)]
    require(len(sub) == len(ordered_ids) and sub.file_id.is_unique,
            f"{condition}/{model}: missing or duplicate rows")
    sub = sub.set_index('file_id').reindex(ordered_ids)
    require(not sub[['generator', 'predicted_generator']].isna().any().any(),
            f"{condition}/{model}: missing values")
    require(np.array_equal(sub.generator.to_numpy(), true), f"{condition}/{model}: label mismatch")
    return sub.predicted_generator.to_numpy(dtype=str)


def point_metrics(truth, pred, classes):
    return {
        'macro_f1': float(f1_score(truth, pred, labels=classes, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(truth, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
    }


def encode(arr, classes):
    encoded = np.searchsorted(classes, np.asarray(arr, dtype=str))
    require((encoded < len(classes)).all(), "Predicted class not present in known classes")
    require(np.array_equal(classes[encoded], np.asarray(arr, dtype=str)), "Predicted class outside classes")
    return encoded


def f1_fast(truth_code, pred_code, n_classes):
    # Identical macro-F1 definition to sklearn with fixed `labels`.
    true_count = np.bincount(truth_code, minlength=n_classes)
    pred_count = np.bincount(pred_code, minlength=n_classes)
    tp = np.bincount(truth_code[truth_code == pred_code], minlength=n_classes)
    denominator = true_count + pred_count
    per_class = np.divide(2.0 * tp, denominator,
                          out=np.zeros(n_classes, dtype=float), where=denominator != 0)
    return float(per_class.mean())


def paired_stratified_bootstrap(true, predictions, conditions, seed, iterations=2000):
    classes = np.sort(np.unique(np.asarray(true, dtype=str)))
    truth_code = encode(true, classes)
    pred_codes = {name: encode(predictions[name], classes) for name in conditions}
    for name in conditions:
        equal_number(f1_fast(truth_code, pred_codes[name], len(classes)),
                     f1_score(true, predictions[name], labels=classes, average='macro', zero_division=0),
                     f"fast F1 equivalence {name}")
    strata = [np.flatnonzero(truth_code == i) for i in range(len(classes))]
    require(len(classes) == 34 and all(len(group) == 20 for group in strata),
            "Bootstrap requires 34 classes x 20 recordings")
    rng = np.random.default_rng(seed)
    values = np.empty((iterations, len(conditions)), dtype=float)
    for i in range(iterations):
        chosen = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        yt = truth_code[chosen]
        for j, name in enumerate(conditions):
            values[i, j] = f1_fast(yt, pred_codes[name][chosen], len(classes))
    return values


def verify_main_and_controls(raw, split_frame):
    base = raw / 'robustness_main_v1'
    selected = load_csv(base / 'selected_test_records.csv')
    check_selection(raw, selected, split_frame)
    selected = selected.sort_values(['generator', 'file_id']).reset_index(drop=True)
    ids = selected.file_id.to_list()
    true = selected.generator.to_numpy(dtype=str)
    classes = np.sort(np.unique(true))
    main = pd.read_parquet(base / 'predictions.parquet')
    require(not main.duplicated(['file_id','condition']).any(), 'Duplicate main predictions')
    require(len(main) == len(ids)*len(MAIN_CONDITIONS), 'Wrong main predictions row count')
    main_pred = {condition: check_predictions(main, ids, true, condition) for condition in MAIN_CONDITIONS}
    saved = load_json(base / 'metrics.json')
    require(set(saved['results']) == set(MAIN_CONDITIONS), 'Missing main metric conditions')
    checks = 0
    clean_f1 = point_metrics(true, main_pred['clean'], classes)['macro_f1']
    for condition, pred in main_pred.items():
        result = point_metrics(true, pred, classes)
        reported = saved['results'][condition]
        for key, value in result.items():
            equal_number(value, reported[key], f"main/{condition}/{key}"); checks += 1
        equal_number(result['macro_f1'] - clean_f1, reported['delta_macro_f1'],
                     f"main/{condition}/delta"); checks += 1
        require(int(reported['n']) == 680, f"main/{condition}/n mismatch")
        require(int(reported['predicted_classes']) == len(set(pred)),
                f"main/{condition}/n predicted mismatch")
    recall = load_csv(base / 'per_generator_recall.csv')
    require(len(recall) == 34*len(MAIN_CONDITIONS) and not recall.duplicated(['condition', 'generator']).any(),
            'per-generator recall row count/keys mismatch')
    for row in recall.itertuples(index=False):
        mask = true == row.generator
        require(row.condition in main_pred and int(mask.sum()) == int(row.n) == 20,
                'per-generator class/condition mismatch')
        score = float(np.mean(main_pred[row.condition][mask] == true[mask]))
        equal_number(score, row.recall, f"recall/{row.condition}/{row.generator}"); checks += 1

    pcm = load_csv(raw / 'robustness_pcm16_control_v1/predictions_comparison.csv')
    safe = load_csv(raw / 'robustness_safe_codec_control_v1/predictions_comparison.csv')
    require(len(pcm) == len(safe) == 680 and pcm.file_id.is_unique and safe.file_id.is_unique,
            'control files missing or duplicate recordings')
    pcm = pcm.set_index('file_id').reindex(ids)
    safe = safe.set_index('file_id').reindex(ids)
    for name, frame in [('pcm', pcm), ('safe', safe)]:
        require(not frame[['generator', 'clean_prediction']].isna().any().any(), f'{name} control missing labels')
        require(np.array_equal(frame.generator.to_numpy(dtype=str), true), f'{name} control labels mismatch')
    mapping = [
        (pcm, 'clean_prediction', 'clean'), (pcm, 'mp3_prediction', 'mp3_64kbps'),
        (safe, 'clean_prediction', 'clean'), (safe, 'mp3_original_prediction', 'mp3_64kbps'),
    ]
    for frame, col, condition in mapping:
        require(np.array_equal(frame[col].to_numpy(dtype=str), main_pred[condition]),
                f'Control prediction mismatch: {condition}/{col}')
        checks += 1
    require(np.array_equal(pcm.pcm16_prediction.to_numpy(dtype=str),
                           safe.pcm16_original_prediction.to_numpy(dtype=str)),
            'PCM16 original prediction mismatch between controls'); checks += 1

    control_mapping = {
        'robustness_pcm16_control_v1': {
            'clean': pcm.clean_prediction.to_numpy(dtype=str),
            'pcm16_only': pcm.pcm16_prediction.to_numpy(dtype=str),
            'mp3_64kbps': pcm.mp3_prediction.to_numpy(dtype=str),
        },
        'robustness_safe_codec_control_v1': {
            'clean': safe.clean_prediction.to_numpy(dtype=str),
            'pcm16_original': safe.pcm16_original_prediction.to_numpy(dtype=str),
            'mp3_original': safe.mp3_original_prediction.to_numpy(dtype=str),
            'pcm16_safe': safe.pcm16_safe_prediction.to_numpy(dtype=str),
            'mp3_safe': safe.mp3_safe_prediction.to_numpy(dtype=str),
        },
    }
    for name, pred_map in control_mapping.items():
        reported = load_csv(raw / name / 'metrics_comparison.csv').set_index('condition')
        require(set(reported.index) == set(pred_map), f'{name}: conditions mismatch')
        for condition, pred in pred_map.items():
            m = point_metrics(true, pred, classes)
            for key in ('macro_f1','accuracy'):
                equal_number(m[key], reported.loc[condition, key], f'{name}/{condition}/{key}'); checks += 1

    bootstrap_predictions = dict(main_pred)
    bootstrap_predictions.update({
        'pcm16_original': safe.pcm16_original_prediction.to_numpy(dtype=str),
        'pcm16_safe': safe.pcm16_safe_prediction.to_numpy(dtype=str),
        'mp3_safe': safe.mp3_safe_prediction.to_numpy(dtype=str),
    })
    conditions = list(bootstrap_predictions)
    boot = paired_stratified_bootstrap(true, bootstrap_predictions, conditions, 20261017)
    pos = {name: i for i, name in enumerate(conditions)}
    metric_rows = load_csv(raw / 'robustness_statistics_v1/macro_f1_bootstrap_ci.csv')
    require(set(metric_rows.condition) == set(conditions) and len(metric_rows) == len(conditions),
            'Bootstrap condition rows mismatch')
    for row in metric_rows.itertuples(index=False):
        require(row.condition in pos, 'Unknown bootstrap condition')
        score = point_metrics(true, bootstrap_predictions[row.condition], classes)
        lo, hi = np.percentile(boot[:, pos[row.condition]], [2.5, 97.5])
        for key in ('macro_f1','accuracy'):
            equal_number(score[key], getattr(row, key), f'bootstrap/{row.condition}/{key}'); checks += 1
        equal_number(lo, row.ci95_low, f'bootstrap/{row.condition}/low'); checks += 1
        equal_number(hi, row.ci95_high, f'bootstrap/{row.condition}/high'); checks += 1
    pairs = load_csv(raw / 'robustness_statistics_v1/paired_macro_f1_differences.csv')
    require(len(pairs) == len(MAIN_COMPARISONS), 'Bootstrap differences count mismatch')
    require(set(zip(pairs.condition_a, pairs.condition_b)) == set(MAIN_COMPARISONS),
            'Bootstrap pair labels mismatch')
    for row in pairs.itertuples(index=False):
        difference = boot[:, pos[row.condition_a]] - boot[:, pos[row.condition_b]]
        score = point_metrics(true, bootstrap_predictions[row.condition_a], classes)['macro_f1'] - \
                point_metrics(true, bootstrap_predictions[row.condition_b], classes)['macro_f1']
        equal_number(score, row.delta_macro_f1, f'paired/{row.condition_a}-{row.condition_b}/delta'); checks += 1
        lo, hi = np.percentile(difference, [2.5, 97.5])
        equal_number(lo, row.ci95_low, f'paired/{row.condition_a}-{row.condition_b}/low'); checks += 1
        equal_number(hi, row.ci95_high, f'paired/{row.condition_a}-{row.condition_b}/high'); checks += 1
    return checks, {c: (ids, true, main_pred[c]) for c in MAIN_CONDITIONS}


def verify_training_seed(raw, seed, frozen, test_reference):
    folder = raw / f'robustness_training_noise30_noise20_n50_seed{seed}'
    meta = load_json(folder / 'metrics.json')
    selection = load_csv(folder / 'selected_train_records.csv')
    require(len(selection) == 1700 and selection.file_id.is_unique,
            f'seed{seed}: expected 1700 unique train selections')
    require(selection.generator.nunique() == 34 and (selection.groupby('generator').size() == 50).all(),
            f'seed{seed}: unexpected train class distribution')
    ids = selection.file_id.to_list()
    require(set(ids).issubset(frozen.index), f'seed{seed}: selected unknown IDs')
    require(frozen.loc[ids, 'split'].eq('train').all(), f'seed{seed}: non-train ID selected')
    require(np.array_equal(frozen.loc[ids, 'generator'].to_numpy(), selection.generator.to_numpy()),
            f'seed{seed}: train labels mismatch')
    protocol = load_json(folder / 'protocol.json')
    require(int(protocol['seed']) == seed and int(protocol['train_parent_records']) == 1700,
            f'seed{seed}: invalid protocol metadata')
    require(meta['protocol'] == protocol, f'seed{seed}: metrics protocol does not match protocol.json')

    predictions = pd.read_parquet(folder / 'test_predictions.parquet')
    require(len(predictions) == 680*len(MAIN_CONDITIONS)*2,
            f'seed{seed}: test_predictions row count mismatch')
    require(not predictions.duplicated(['file_id','condition','model']).any(),
            f'seed{seed}: duplicate condition/model/ID')
    summary = load_csv(folder / 'condition_summary.csv')
    require(len(summary) == 12 and not summary.duplicated(['condition','model']).any(),
            f'seed{seed}: summary missing/duplicate rows')
    saved_summary = {(r['condition'],r['model']):r for r in meta['summary']}
    expected_pairs = {(condition, model) for condition in MAIN_CONDITIONS
                      for model in ('clean_trained','augmented_trained')}
    require(set(saved_summary) == set(zip(summary.condition,summary.model)) == expected_pairs,
            f'seed{seed}: condition/model sets do not agree')
    check_count = 0
    preds_by_condition = {}
    for condition in MAIN_CONDITIONS:
        ids_ref, true_ref, baseline = test_reference[condition]
        # Original train-script bootstrap sorts by file_id (not generator+ID).
        order = np.argsort(np.asarray(ids_ref, dtype=str))
        ordered_ids = np.asarray(ids_ref)[order].tolist()
        true = true_ref[order]
        classes = np.sort(np.unique(true))
        for model in ('clean_trained','augmented_trained'):
            pred = check_predictions(predictions, ordered_ids, true, condition, model=model)
            if model == 'clean_trained':
                require(np.array_equal(pred, baseline[order]),
                        f'seed{seed}/{condition}: baseline predictions changed')
            metrics = point_metrics(true, pred, classes)
            saved_csv = summary.set_index(['condition','model']).loc[(condition,model)]
            saved_json = saved_summary[(condition,model)]
            for key in ('macro_f1','accuracy','balanced_accuracy'):
                equal_number(metrics[key], saved_csv[key], f'seed{seed}/{condition}/{model}/csv/{key}')
                equal_number(metrics[key], saved_json[key], f'seed{seed}/{condition}/{model}/json/{key}')
                check_count += 2
            require(int(saved_csv['n']) == int(saved_json['n']) == 680,
                    f'seed{seed}/{condition}/{model}: wrong n')
            preds_by_condition[(condition, model)] = pred

    require(int(meta['bootstrap_repetitions']) == 2000 and int(meta['bootstrap_seed']) == 42,
            f'seed{seed}: bootstrap settings mismatch')
    saved_pairs = load_csv(folder / 'paired_robustness_bootstrap.csv')
    require(set(saved_pairs.condition) == set(MAIN_CONDITIONS) and len(saved_pairs) == 6,
            f'seed{seed}: paired bootstrap CSV rows mismatch')
    saved_pairs_json = {row['condition']: row for row in meta['paired_bootstrap']}
    require(set(saved_pairs_json) == set(MAIN_CONDITIONS), f'seed{seed}: paired bootstrap JSON rows mismatch')

    # Match the original algorithm: seed reset once, then six condition-specific
    # sequences of 2000 stratified resamples, preserving condition order.
    rng = np.random.default_rng(42)
    for condition in MAIN_CONDITIONS:
        ref_ids, ref_true, _ = test_reference[condition]
        order = np.argsort(np.asarray(ref_ids, dtype=str))
        true = ref_true[order]
        classes = np.sort(np.unique(true))
        y = encode(true, classes)
        pred_clean = preds_by_condition[(condition,'clean_trained')]
        pred_aug = preds_by_condition[(condition,'augmented_trained')]
        clean_c = encode(pred_clean, classes)
        aug_c = encode(pred_aug, classes)
        strata = [np.flatnonzero(y == i) for i in range(len(classes))]
        require(all(len(g)==20 for g in strata), f'seed{seed}/{condition}: invalid class strata')
        boots = np.empty((2000,3), dtype=float)
        for i in range(2000):
            idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            a = f1_fast(y[idx], clean_c[idx], len(classes))
            b = f1_fast(y[idx], aug_c[idx], len(classes))
            boots[i] = (a, b, b-a)
        expected = {
            'clean_trained_macro_f1': f1_fast(y, clean_c, len(classes)),
            'augmented_trained_macro_f1': f1_fast(y, aug_c, len(classes)),
            'delta_macro_f1_aug_minus_clean': f1_fast(y, aug_c, len(classes)) - f1_fast(y, clean_c, len(classes)),
            'clean_trained_accuracy': float(np.mean(pred_clean == true)),
            'augmented_trained_accuracy': float(np.mean(pred_aug == true)),
        }
        saved_row = saved_pairs.set_index('condition').loc[condition]
        saved_json = saved_pairs_json[condition]
        for key, value in expected.items():
            equal_number(value, saved_row[key], f'seed{seed}/{condition}/paired_csv/{key}')
            equal_number(value, saved_json[key], f'seed{seed}/{condition}/paired_json/{key}')
            check_count += 2
        for key, j in (
            ('clean_macro_f1_95ci',0),('aug_macro_f1_95ci',1),('delta_macro_f1_95ci',2)
        ):
            ci = np.percentile(boots[:,j], [2.5,97.5]).tolist()
            compare_ci(ci, saved_row[key], f'seed{seed}/{condition}/paired_csv/{key}')
            compare_ci(ci, saved_json[key], f'seed{seed}/{condition}/paired_json/{key}')
            check_count += 4
    return check_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    out = args.output_dir.resolve()
    require(out != root and root not in out.parents,
            '--output-dir must be outside --project-root (read-only release)')
    raw = root / 'results/raw'
    require(raw.is_dir(), f'Missing directory: {raw}')
    out.mkdir(parents=True, exist_ok=True)
    print('Source project (READ ONLY):', root, flush=True)
    print('Reports outside project:', out, flush=True)

    checks = []
    errors = []
    frozen = pd.read_csv(root / 'splits/closed_set.csv', usecols=['file_id','generator','split'], dtype={'file_id':str})
    require(frozen.file_id.is_unique, 'Frozen split contains duplicate recording IDs')
    frozen_indexed = frozen.set_index('file_id')
    try:
        count, test_reference = verify_main_and_controls(raw, frozen)
        checks.append({'scope':'robustness_main_controls_bootstrap', 'checks':count,'status':'PASS'})
        print(f'Main robustness + codec controls + 2000-bootstrap: PASS ({count} numerical checks)', flush=True)
    except Exception as exc:
        errors.append('main robustness: ' + str(exc))
        print('Main robustness: FAIL:', exc, flush=True)
        test_reference = None

    if test_reference is not None:
        for seed in SEEDS:
            try:
                count = verify_training_seed(raw, seed, frozen_indexed, test_reference)
                checks.append({'scope':f'robustness_training_seed{seed}', 'checks':count,'status':'PASS'})
                print(f'Training robustness seed {seed}: PASS ({count} numerical checks)', flush=True)
            except Exception as exc:
                errors.append(f'seed {seed}: {exc}')
                print(f'Training robustness seed {seed}: FAIL: {exc}', flush=True)

    report = {'project':str(root), 'scope':'Frozen-predictions robustness numerical verification',
              'method':'independent macro-F1/accuracy/recall and seeded stratified paired bootstrap',
              'checks':checks, 'errors':errors,
              'note':'Does not re-run audio perturbations, feature extraction or model training.'}
    target = out/'robustness_verification.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\n=== FINAL VERIFICATION ===', flush=True)
    print('Scopes PASS:', len(checks), '/ 6', flush=True)
    print('Numerical checks PASS:', sum(x['checks'] for x in checks), flush=True)
    print('Errors:',len(errors), flush=True)
    print('Report:',target, flush=True)
    print('FINAL STATUS:', 'PASS' if len(checks)==6 and not errors else 'REVIEW REQUIRED', flush=True)
    return 0 if len(checks)==6 and not errors else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FATAL:', type(exc).__name__, str(exc), flush=True)
        traceback.print_exc()
        sys.exit(1)
