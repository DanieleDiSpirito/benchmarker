#!/usr/bin/env python3
"""
Step 1: Dataset Sanitizer & Ground-Truth Verifier
- Reads data/tier1.json .. data/tier4.json (auto-strips trailing quotes/markdown fences)
- Compiles every variant (classic_c, randomized_c, strange_c) under:
    1) gcc -std=c11 -O0 -Wall -Wextra -Werror -fsanitize=undefined,address
    2) clang -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=undefined,address
- Verifies zero compiler warnings, zero sanitizer errors, identical stdout across
  both compilers, and identical stdout across all 3 variants.
- Saves verified triplets with ground-truth stdout to results/validated_dataset.json.
"""

import glob
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUT_FILE = pathlib.Path(__file__).parent / "results" / "validated_dataset.json"
VARIANTS = ("classic_c", "randomized_c", "strange_c")


def load_clean_json(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8").strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Strip stray leading/trailing quotes from web chat copy-paste
    if raw.startswith('"') and not raw.startswith('"{'):
        raw = raw[1:]
    if raw.endswith('"') and not raw.endswith('}"'):
        raw = raw[:-1]
    # Trim anything before first '{' or after last '}'
    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last != -1:
        raw = raw[first : last + 1]
    return json.loads(raw)


def compile_and_run(c_code: str, compiler: str, opt: str) -> tuple[bool, str]:
    if shutil.which(compiler) is None:
        return False, f"Compiler not found: {compiler}"
    with tempfile.TemporaryDirectory(prefix="c_val_") as tmpdir:
        src = pathlib.Path(tmpdir) / "prog.c"
        exe = pathlib.Path(tmpdir) / "prog"
        src.write_text(c_code, encoding="utf-8")
        cmd = [
            compiler,
            "-std=c11",
            opt,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined,address",
            "-fno-sanitize-recover=all",
            str(src),
            "-o",
            str(exe),
        ]
        comp = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if comp.returncode != 0:
            return False, f"{compiler} {opt} compile error: {comp.stderr.strip()}"
        try:
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=3)
        except subprocess.TimeoutExpired:
            return False, f"{compiler} {opt} execution timeout (>3s)"
        if run.returncode != 0:
            return False, f"{compiler} {opt} runtime/sanitizer error: {run.stderr.strip()}"
        return True, run.stdout.strip()


def main() -> None:
    files = sorted(DATA_DIR.glob("tier*.json"))
    if not files:
        print(f"[ERROR] No tier*.json files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    has_clang = shutil.which("clang") is not None
    if shutil.which("gcc") is None:
        print("[ERROR] gcc is required for validation.", file=sys.stderr)
        sys.exit(1)
    if not has_clang:
        print("[WARN] clang not found; validating with gcc -O0 and gcc -O2 only.")

    validated = []
    rejected = []

    for fpath in files:
        payload = load_clean_json(fpath)
        file_tier = payload.get("tier")
        for prob in payload.get("problems", []):
            pid = prob.get("id", "unknown")
            tier = prob.get("tier", file_tier)
            outputs = {}
            ok_all = True
            reason = ""

            for var in VARIANTS:
                code = prob.get(var, "")
                if not code.strip():
                    ok_all = False
                    reason = f"Missing {var}"
                    break

                ok1, out1 = compile_and_run(code, "gcc", "-O0")
                if not ok1:
                    ok_all = False
                    reason = f"[{var}] {out1}"
                    break

                second_cc = "clang" if has_clang else "gcc"
                ok2, out2 = compile_and_run(code, second_cc, "-O2")
                if not ok2:
                    ok_all = False
                    reason = f"[{var}] {out2}"
                    break

                if out1 != out2:
                    ok_all = False
                    reason = f"[{var}] Optimizer mismatch: gcc -O0='{out1}' vs {second_cc} -O2='{out2}'"
                    break

                outputs[var] = out1

            if ok_all and len(set(outputs.values())) != 1:
                ok_all = False
                reason = f"Variant stdout mismatch: {outputs}"

            if ok_all:
                validated.append(
                    {
                        "id": pid,
                        "tier": int(tier),
                        "algorithm_concept": prob.get("algorithm_concept", ""),
                        "complexity_tags": prob.get("complexity_tags", []),
                        "ground_truth_stdout": outputs["classic_c"],
                        "classic_c": prob["classic_c"],
                        "randomized_c": prob["randomized_c"],
                        "strange_c": prob["strange_c"],
                    }
                )
            else:
                rejected.append({"id": pid, "tier": tier, "reason": reason})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(
            {
                "total_validated": len(validated),
                "total_rejected": len(rejected),
                "problems": validated,
                "rejected": rejected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Validation complete: {len(validated)} triplets passed, {len(rejected)} rejected.")
    for r in rejected:
        print(f"  [REJECTED] {r['id']}: {r['reason']}")
    print(f"Saved validated dataset to {OUT_FILE}")


if __name__ == "__main__":
    main()
