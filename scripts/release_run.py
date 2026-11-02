import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "f1ae86e0ebc76c6f6ddf3fe349844347ada5fc460be496edc8087c73f493f6e0": "RuASD split",
    "cac3bf00c0e8f77a1a2a37a4b3b005c1ce65ba3dbaf364d3b028dd5dde20171d": "RuASD folds",
    "f4b66a0c8cbce8e7629d85838eb371d5f7381a9d2eafb63917f8ccf1fe5fbd66": "MLAAD split",
    "eac341ab00a413ff1360ed4324316ff7822a056c4d02717e62b6c9691a66cde0": "MLAAD folds",
}

def load_config():
    path = ROOT / "configs" / "local_paths.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))

def resolve_data_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()

def verify_splits():
    found = {}
    directory = ROOT / "splits"
    if directory.exists():
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in EXPECTED:
                found[digest] = path.relative_to(ROOT).as_posix()

    print("\nFrozen splits:")
    for digest, label in EXPECTED.items():
        print(f"  {label}: {'OK' if digest in found else 'MISSING'}")
    return len(found)

def verify_python():
    passed = 0
    errors = []
    for dirname in ("scripts", "src", "tests"):
        folder = ROOT / dirname
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            try:
                source = path.read_text(encoding="utf-8-sig")
                compile(source, str(path), "exec")
                passed += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    print(f"\nPython syntax: {passed} passed, {len(errors)} failed")
    for error in errors[:20]:
        print("  ERROR:", error)
    return errors

def main():
    parser = argparse.ArgumentParser(
        description="Portable experiment runner and release checks"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ruasd-root")
    parser.add_argument("--mlaad-root")
    parser.add_argument("--script")
    args, extra = parser.parse_known_args()

    cfg = load_config()
    ru = (
        args.ruasd_root
        or os.environ.get("RUASD_ROOT")
        or cfg.get("ruasd_root")
        or "data/raw/RuASD"
    )
    ml = (
        args.mlaad_root
        or os.environ.get("MLAAD_ROOT")
        or cfg.get("mlaad_root")
        or "data/external/MLAAD_fake_ru"
    )

    ru = resolve_data_path(ru)
    ml = resolve_data_path(ml)

    print("Project:", ROOT)
    print("RuASD:", ru, "| exists:", ru.is_dir())
    print("MLAAD:", ml, "| exists:", ml.is_dir())

    if args.check:
        matches = verify_splits()
        errors = verify_python()
        print("\nRelease structural check:",
              "PASS" if matches == 4 and not errors else "FAIL")
        return 0 if matches == 4 and not errors else 1

    if not args.script:
        parser.error("Specify --check or --script")

    script = (ROOT / args.script).resolve()
    if not script.is_file() or not script.is_relative_to(ROOT):
        parser.error("Script must be an existing file inside the project")

    env = os.environ.copy()
    env["RUASD_ROOT"] = str(ru)
    env["MLAAD_ROOT"] = str(ml)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    )

    if extra and extra[0] == "--":
        extra = extra[1:]

    command = [sys.executable, str(script), *extra]
    print("Running:", script.relative_to(ROOT))
    return subprocess.call(command, cwd=str(ROOT), env=env)

if __name__ == "__main__":
    raise SystemExit(main())
