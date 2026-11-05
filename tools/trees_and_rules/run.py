"""Fit optimal sparse trees (GOSDT) and rule lists (CORELS) on the Gray-code bits of the solubility tiers.

Runs in the Python 3.9 side environment (see setup.sh); the notebook calls it and reads the files back.
One row per compound, labelled with its consensus tier's two Gray bits; features binarized by threshold guessing on the
train side of each split. For every (split, bit, regularization) both solvers are fitted on the train side and every
compound gets the training frequency of bit = 1 in the leaf (or rule) it falls in, with +0.5 Laplace smoothing.

Usage: run.py --root <repo> --splits cv_0 cv_1 ... --regularization 0.005 0.02 --out <dir>
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from corels import CorelsClassifier  # noqa: E402
from gosdt import GOSDTClassifier, ThresholdGuessBinarizer  # noqa: E402

GRAY = {"bit_1": (2, 3), "bit_2": (1, 2)}  # tiers in which the bit is 1: 0->00, 1->01, 2->11, 3->10
BINARIZER = dict(n_estimators=40, max_depth=1, random_state=0)  # one threshold per boosting stump, at most 40
GOSDT = dict(depth_budget=5, time_limit=60, verbose=False)
CORELS = dict(max_card=2, n_iter=1_000_000, verbosity=[])


LEFT = 1  # the feature value that goes to a GOSDT node's left child; gosdt_leaf checks it against the tree's own predict


def leaf_paths(node, path=()):
    """Every leaf of a GOSDT tree as (path of (feature, value), prediction)."""
    if hasattr(node, "feature"):
        return (leaf_paths(node.left_child, path + ((node.feature, LEFT),))
                + leaf_paths(node.right_child, path + ((node.feature, 1 - LEFT),)))
    return [(path, int(node.prediction))]


def gosdt_leaf(tree, xb: np.ndarray) -> np.ndarray:
    """Index of the leaf each row falls in; checked against the tree's own predictions."""
    leaves = leaf_paths(tree.tree)
    idx = np.full(len(xb), -1)
    for k, (path, _) in enumerate(leaves):
        mask = np.ones(len(xb), bool)
        for feature, value in path:
            mask &= xb[:, feature] == value
        idx[mask] = k
    assert (idx >= 0).all(), "a row reached no leaf"
    own = np.asarray(tree.predict(xb)).astype(int).ravel()
    assert (np.array([leaves[k][1] for k in idx]) == own).all(), "leaf traversal disagrees with the tree's predict (LEFT convention)"
    return idx


def corels_rule(rules, xb: np.ndarray) -> np.ndarray:
    """Index of the first rule whose antecedents hold for each row (the last rule is the default)."""
    idx = np.full(len(xb), -1)
    for k, rule in enumerate(rules):
        mask = idx < 0
        for a in rule["antecedents"]:
            if a == 0:
                continue  # the default rule
            mask &= (xb[:, abs(a) - 1] == 1) if a > 0 else (xb[:, abs(a) - 1] == 0)
        idx[mask] = k
    assert (idx >= 0).all(), "a row matched no rule"
    return idx


def smoothed(groups: np.ndarray, labels: np.ndarray, train: np.ndarray, n_groups: int) -> np.ndarray:
    """Training frequency of label = 1 per group, +0.5 smoothing, mapped back to every row."""
    ones = np.bincount(groups[train], weights=labels[train], minlength=n_groups)
    size = np.bincount(groups[train], minlength=n_groups)
    return ((ones + 0.5) / (size + 1.0))[groups]


def tree_text(tree, names) -> str:
    lines = []

    def walk(node, depth):
        if hasattr(node, "feature"):
            yes, no = (node.left_child, node.right_child) if LEFT == 1 else (node.right_child, node.left_child)
            lines.append("  " * depth + f"if {names[node.feature]}:")
            walk(yes, depth + 1)
            lines.append("  " * depth + "else:")
            walk(no, depth + 1)
        else:
            lines.append("  " * depth + f"-> {int(node.prediction)}")

    walk(tree.tree, 0)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--regularization", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(root / "data" / "aqsoldb_features.csv", index_col="InChIKey")
    votes = pd.read_csv(root / "data" / "aqsoldb_tiers.csv")
    tier = votes.groupby("InChIKey")["consensus_tier"].first().reindex(features.index)
    assert tier.notna().all(), "a compound has no consensus tier"
    splits = pd.read_csv(root / "data" / "aqsoldb_splits.csv", index_col="InChIKey").reindex(features.index)
    ls = pd.read_csv(root / "data" / "aqsoldb_splits_ls.csv", index_col="InChIKey").reindex(features.index)
    sides = splits.join(ls[[c for c in ls.columns if c.startswith("ls_") and c[3:].isdigit()]])
    bits = {b: tier.isin(t).to_numpy().astype(int) for b, t in GRAY.items()}
    names = list(features.columns)

    predictions, models = {}, {}
    started = time.time()
    for split in args.splits:
        side = sides[split].to_numpy()
        train = side == "train"
        fill = features[train].median()
        x = features.fillna(fill)
        for bit, label in bits.items():
            enc = ThresholdGuessBinarizer(**BINARIZER).fit(x[train], label[train])
            xb = np.asarray(enc.transform(x)).astype(np.uint8)
            j0, v0 = enc.thresholds_[0]  # the binarizer's columns are "descriptor <= threshold"; checked, not assumed
            assert (xb[:, 0] == (x.iloc[:, j0].to_numpy() <= v0)).all(), "binarized column is not x <= threshold"
            thresholds = [f"{names[j]} <= {v:.4g}" for j, v in enc.thresholds_]
            for reg in args.regularization:
                key = f"{split}|{bit}|{reg}"
                # GOSDT
                t0 = time.time()
                clf = GOSDTClassifier(regularization=reg, **GOSDT).fit(xb[train], label[train])
                tree = clf.trees_[0]
                leaf = gosdt_leaf(tree, xb)
                predictions["gosdt", key] = smoothed(leaf, label, train, len(leaf_paths(tree.tree)))
                models["gosdt", key] = {"text": tree_text(tree, thresholds), "leaves": len(leaf_paths(tree.tree)),
                                        "depth": max(len(p) for p, _ in leaf_paths(tree.tree)),
                                        "features used": len({f for p, _ in leaf_paths(tree.tree) for f, _ in p}),
                                        "seconds": round(time.time() - t0, 1), "thresholds": len(thresholds),
                                        "optimal": str(getattr(clf.result_, "status", ""))}
                # CORELS
                t0 = time.time()
                c = CorelsClassifier(c=reg, **CORELS).fit(xb[train], label[train], features=thresholds)
                rules = c.rl().rules
                fired = corels_rule(rules, xb)
                assert (np.array([int(rules[k]["prediction"]) for k in fired]) == np.asarray(c.predict(xb)).astype(int)).all(), \
                    "rule traversal disagrees with CORELS' predict"
                predictions["corels", key] = smoothed(fired, label, train, len(rules))
                models["corels", key] = {"text": str(c.rl()), "rules": len(rules),
                                         "antecedents": sum(len([a for a in r["antecedents"] if a]) for r in rules),
                                         "seconds": round(time.time() - t0, 1), "thresholds": len(thresholds)}
                print(f"{key:22s} gosdt {models['gosdt', key]['leaves']:2d} leaves {models['gosdt', key]['seconds']:5.1f}s | "
                      f"corels {models['corels', key]['rules']:2d} rules {models['corels', key]['seconds']:5.1f}s | {time.time() - started:.0f}s", flush=True)

    for method in ("gosdt", "corels"):
        frame = pd.DataFrame({k: v for (m, k), v in predictions.items() if m == method}, index=features.index)
        path = out / f"predictions_{method}.csv"
        if path.exists():  # a later phase adds columns to an earlier one
            frame = pd.read_csv(path, index_col="InChIKey").reindex(features.index).drop(columns=frame.columns, errors="ignore").join(frame)
        frame.round(4).to_csv(path)
    meta_path = out / "models.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update({f"{m}|{k}": v for (m, k), v in models.items()})
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"wrote {len(predictions)} prediction columns and {len(models)} models to {out}")


if __name__ == "__main__":
    main()
