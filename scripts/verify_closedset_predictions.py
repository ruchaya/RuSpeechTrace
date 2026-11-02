import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUTPUT = ROOT / "reproducibility" / "verification"

EXPERIMENTS = [
    {
        "name": "TADA_RuASD",
        "file": "tada_w2vbert2_layer4_knn21/ruasd/closed_set/test_predictions.csv",
        "id": "id",
        "true": "true_generator",
        "pred": "pred_generator",
        "rows": 13600,
        "classes": 34,
        "expected_f1": 0.9042,
        "expected_accuracy": 0.9042,
    },
    {
        "name": "TADA_MLAAD",
        "file": "tada_w2vbert2_layer4_knn21/mlaad/closed_set/test_predictions.csv",
        "id": "id",
        "true": "true_generator",
        "pred": "pred_generator",
        "rows": 4600,
        "classes": 23,
        "expected_f1": 0.7998,
        "expected_accuracy": None,
    },
    {
        "name": "AASIST_seed42",
        "file": "aasist_fixed4_ruasd34_seed42/test_predictions.parquet",
        "id": "file_id",
        "true": "generator_true",
        "pred": "generator_pred",
        "rows": 13600,
        "classes": 34,
        "expected_f1": 0.9114,
        "expected_accuracy": 0.9144,
    },
    {
        "name": "Acoustic_WavLM_fusion",
        "file": "fixed4_acoustic_wavlm_late_fusion_saved/predictions.parquet",
        "id": "file_id",
        "true": "generator_true",
        "pred": "generator_pred",
        "rows": 13600,
        "classes": 34,
        "expected_f1": 0.9661,
        "expected_accuracy": 0.9662,
    },
]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def matches(value, expected):
    if expected is None:
        return None
    return abs(value - expected) <= 0.00005001

def evaluate(spec):
    path = RAW / spec["file"]

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = [spec["id"], spec["true"], spec["pred"]]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{spec['name']}: missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError(f"{spec['name']}: null IDs or labels")

    if df[spec["id"]].duplicated().any():
        raise ValueError(f"{spec['name']}: duplicate recording IDs")

    true = df[spec["true"]].astype(str).to_numpy()
    pred = df[spec["pred"]].astype(str).to_numpy()

    true_classes = set(true)
    predicted_classes = set(pred)

    if not predicted_classes.issubset(true_classes):
        raise ValueError(
            f"{spec['name']}: predicted labels not present "
            f"in the true class set: {predicted_classes - true_classes}"
        )

    accuracy = float(accuracy_score(true, pred))
    macro_f1 = float(f1_score(true, pred, average="macro"))

    checks = {
        "rows": len(df) == spec["rows"],
        "classes": len(true_classes) == spec["classes"],
        "macro_f1": matches(macro_f1, spec["expected_f1"]),
    }

    if spec["expected_accuracy"] is not None:
        checks["accuracy"] = matches(
            accuracy, spec["expected_accuracy"]
        )

    passed = all(checks.values())

    return {
        "experiment": spec["name"],
        "source": ("results/raw/" + spec["file"]).replace("\\", "/"),
        "source_sha256": sha256_file(path),
        "rows": int(len(df)),
        "n_classes": int(len(true_classes)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "paper_macro_f1": spec["expected_f1"],
        "paper_accuracy": spec["expected_accuracy"],
        "checks": checks,
        "status": "PASS" if passed else "FAIL",
    }

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []

    for spec in EXPERIMENTS:
        print(f"\n=== {spec['name']} ===", flush=True)
        try:
            result = evaluate(spec)
            results.append(result)

            print("Records:", result["rows"])
            print("Classes:", result["n_classes"])
            print(f"Accuracy: {result['accuracy']:.8f}")
            print(f"Macro-F1: {result['macro_f1']:.8f}")
            print("Expected Macro-F1:", result["paper_macro_f1"])
            print("Checks:", result["checks"])
            print("STATUS:", result["status"])

            if result["status"] != "PASS":
                errors.append(spec["name"] + ": mismatch")

        except Exception as exc:
            errors.append(f"{spec['name']}: {exc}")
            print("STATUS: ERROR")
            print("ERROR:", exc)

    metadata = {
        "purpose": "Independent closed-set prediction metric verification",
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "averaging": "macro",
        "metric_library": "sklearn.metrics",
        "tolerance": 0.00005001,
        "results": results,
        "errors": errors,
    }

    json_path = OUTPUT / "closed_set_metrics_recomputed.json"
    csv_path = OUTPUT / "closed_set_metrics_recomputed.csv"

    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        {
            key: value
            for key, value in result.items()
            if key != "checks"
        }
        for result in results
    ]

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n================================")
    print("FINAL VERIFICATION")
    print("Experiments checked:", len(results))
    print("Passed:", sum(x["status"] == "PASS" for x in results))
    print("Errors:", len(errors))
    print("JSON:", json_path.relative_to(ROOT))
    print("CSV:", csv_path.relative_to(ROOT))

    if errors or len(results) != len(EXPERIMENTS):
        print("FINAL STATUS: FAIL")
        return 1

    print("FINAL STATUS: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
