"""Reproducibility check: compare two full generation runs made with the same prompts and seeds.

Run 1 is kept in data/runs/run1/ (raw generations + labelled mentions); run 2 is the current data/.
Writes results/run_comparison.json and summary.json["reproducibility"] (aggregates only; the ghost
names that differ go to private/run_diff_ghosts.csv). Needs the private data/ cache of both runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from phantom import config, runconfig

RUN1 = config.DATA / "runs" / "run1"
OUT = config.RESULTS / "run_comparison.json"
PRIVATE_DIFF = config.PRIVATE / "run_diff_ghosts.csv"


def local_records(path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["key"]: r for r in rows if r.get("arm") != "hosted"}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    r1, r2 = local_records(RUN1 / "raw_generations.jsonl"), local_records(config.RAW_GENERATIONS_PATH)
    common = sorted(set(r1) & set(r2))
    same, total = Counter(), Counter()
    for k in common:
        m = r1[k]["model"]
        total[m] += 1
        same[m] += r1[k]["output"] == r2[k]["output"]
    l1, l2 = pd.read_parquet(RUN1 / "labelled.parquet"), pd.read_parquet(config.DATA / "labelled.parquet")
    g1, g2 = set(l1.loc[l1["y_ghost"], "name"]), set(l2.loc[l2["y_ghost"], "name"])
    result = {
        "generated_at": runconfig.utc_now(),
        "setup": "same 200 prompts, 3 models, seeds 1001/1002, temperature 0.7, Ollama; run 2 started from scratch",
        "generations": {"run1": len(r1), "run2": len(r2), "common_keys": len(common),
                        "identical_outputs": sum(same.values()),
                        "identical_share": sum(same.values()) / len(common),
                        "per_model": {m: {"identical": same[m], "total": total[m]} for m in sorted(total)}},
        "ghosts": {"mentions_run1": int(l1["y_ghost"].sum()), "mentions_run2": int(l2["y_ghost"].sum()),
                   "names_run1": len(g1), "names_run2": len(g2), "names_in_both": len(g1 & g2),
                   "only_run1": len(g1 - g2), "only_run2": len(g2 - g1)},
        "generation_hours": {"run1": sum(r["latency_s"] for r in r1.values()) / 3600,
                             "run2": sum(r["latency_s"] for r in r2.values()) / 3600},
    }
    runconfig.save(result, OUT)
    config.PRIVATE.mkdir(exist_ok=True)
    pd.DataFrame([{"name": n, "run": "run1_only"} for n in sorted(g1 - g2)] +
                 [{"name": n, "run": "run2_only"} for n in sorted(g2 - g1)]).to_csv(PRIVATE_DIFF, index=False)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["reproducibility"] = result
    runconfig.save(summary, config.SUMMARY_PATH)
    g = result["generations"]
    print(f"identical outputs {g['identical_outputs']}/{g['common_keys']} ({g['identical_share']:.1%}); "
          f"ghost names run1 {len(g1)}, run2 {len(g2)}, both {len(g1 & g2)}")


if __name__ == "__main__":
    main()
