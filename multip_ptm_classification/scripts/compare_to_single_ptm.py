"""
Compare multi-PTM evaluation JSON to single-PTM classifier runs.

Single-PTM layout (default): classifier_runs/<ptm>/<model_type>/<run>/metrics_test.json
Multi-PTM: metrics_test_per_ptm.json produced by eval_multip.py
"""

import argparse
import json
import os
from typing import Any, Dict, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _find_latest_metrics_test(ptm: str, model_type: str, classifier_root: str) -> Optional[str]:
    base = os.path.join(classifier_root, ptm, model_type)
    if not os.path.isdir(base):
        return None
    candidates = []
    for run in os.listdir(base):
        p = os.path.join(base, run, "metrics_test.json")
        if os.path.isfile(p):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def main():
    p = argparse.ArgumentParser(description="Tabulate multip vs single-PTM metrics.")
    p.add_argument("--multip_json", required=True, help="eval_multip metrics_test_per_ptm.json")
    p.add_argument("--classifier_root", default="classifier_runs")
    p.add_argument("--single_model_type", default="cnn_gru")
    args = p.parse_args()

    multip = _load_json(args.multip_json)
    per = multip.get("per_ptm", {})
    keys = ["mcc", "f1", "aupr", "auroc"]
    print(f"{'PTM':<16} {'multi_MCC':>10} {'single_MCC':>11} {'multi_F1':>9} {'single_F1':>10} "
          f"{'multi_AUPR':>10} {'single_AUPR':>11} {'multi_AUC':>9} {'single_AUC':>10}")
    for name in sorted(per.keys()):
        mrow = per[name]
        spath = _find_latest_metrics_test(name, args.single_model_type, args.classifier_root)
        if spath:
            srow = _load_json(spath)
        else:
            srow = {}
        def g(d, k):
            v = d.get(k)
            return float(v) if v is not None and v == v else float("nan")
        print(
            f"{name:<16} "
            f"{g(mrow,'mcc'):10.4f} {g(srow,'mcc'):11.4f} "
            f"{g(mrow,'f1'):9.4f} {g(srow,'f1'):10.4f} "
            f"{g(mrow,'aupr'):10.4f} {g(srow,'aupr'):11.4f} "
            f"{g(mrow,'auroc'):9.4f} {g(srow,'auroc'):10.4f}"
            + ("" if spath else "  (no single run)")
        )


if __name__ == "__main__":
    main()
