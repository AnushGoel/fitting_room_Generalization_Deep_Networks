"""Command-line entry point: python -m fitting_room <command>."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fitting-room", description="Generalization study on Fashion-MNIST")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train configurations headlessly into the shared run cache")
    t.add_argument("--config", default="configs/experiments.toml")
    t.add_argument("--only", default="", help="comma-separated experiment keys (default: all)")
    t.add_argument("--seeds", choices=["main", "all"], default="main", help="main seed only, or main plus extra seeds")
    t.add_argument("--force", action="store_true", help="retrain even if a cached run exists")

    for name, helptext in [("registry", "rebuild artifacts/registry.csv from the run cache"),
                           ("hypotheses", "print the pre-registered hypothesis scorecard"),
                           ("report", "write docs/RESULTS.md, compressed figures, and refresh the README results block"),
                           ("slim", "write a small, deploy-ready copy of the artifacts")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--artifacts", default="artifacts")
        if name == "report":
            p.add_argument("--docs", default="docs")
            p.add_argument("--readme", default="README.md")
        if name == "slim":
            p.add_argument("--out", default="artifacts_slim")
            p.add_argument("--n-val", type=int, default=2000)
            p.add_argument("--budget-mb", type=float, default=None)

    a = ap.parse_args(argv)
    if a.cmd == "train":
        from .train import train_from_config
        train_from_config(a.config, only=[k for k in a.only.split(",") if k], all_seeds=a.seeds == "all", force=a.force)
    elif a.cmd == "registry":
        from .artifacts import write_registry
        print("wrote", write_registry(a.artifacts))
    elif a.cmd == "hypotheses":
        from .artifacts import load
        from .hypotheses import evaluate
        for r in evaluate(load(a.artifacts)):
            print(f"{r.hypothesis.id:>4}  {r.verdict:<14} {r.hypothesis.topic}: {r.evidence}")
    elif a.cmd == "report":
        from .artifacts import load
        from .report import write_all
        info = write_all(load(a.artifacts), Path(a.docs), Path(a.readme))
        print(f"wrote {a.docs}/RESULTS.md; {info['figures']} figures "
              f"({info['fig_mb_before']:.1f} MB -> {info['fig_mb_after']:.1f} MB); README updated: {info['readme_updated']}")
    elif a.cmd == "slim":
        from .slim import slim
        info = slim(a.artifacts, a.out, n_val=a.n_val, budget_mb=a.budget_mb)
        print(f"slim bundle: {info['before_mb']:.1f} MB -> {info['after_mb']:.1f} MB "
              f"({info['n_val']} validation images, {info['n_test_errors']} test errors kept) in {a.out}")


if __name__ == "__main__":
    main()
