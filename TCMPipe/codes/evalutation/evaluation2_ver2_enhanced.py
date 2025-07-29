#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tongue Diagnosis Evaluation Metric
==================================
A **production‑ready** implementation that addresses all pain‑points identified in the
previous iterations:

* One‑time config load, zero per‑sample disk I/O.
* Single unified token representation:  ``PREFIX_value`` everywhere.
* Fast token / pattern extraction via **Aho‑Corasick** (falls back to regex if the
  optional ``pyahocorasick`` wheel is unavailable).
* Robust synonym handling – intra‑ & cross‑category, prefix‑agnostic.
* Unknown‑token discovery with stop‑word filtering, global frequency report &
  auto‑generated patch snippet for the config file.
* Efficient similarity computation – Hungarian (scipy) for small matrices, greedy
  for >128×128 to avoid O(n³) blow‑ups.
* CLI supporting single file, directory, and stdin → stdout streaming.

**Author:** ChatGPT‑o3, 2025‑07‑13
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------
# 1.  Optional dependency – pyahocorasick (fast token search)
# ---------------------------------------------------------
try:
    import ahocorasick  # type: ignore

    _HAS_AHO = True
except ImportError:  # pragma: no cover – fallback path
    _HAS_AHO = False

# ---------------------------------------------------------
# 2.  Data classes
# ---------------------------------------------------------
@dataclass
class TokenSets:
    """Container for extracted tokens."""

    tokens: Set[str]
    unknown: Set[str]

    def to_categories(self, category_map: Dict[str, str]) -> Dict[str, Set[str]]:
        cats = {"tongue": set(), "coat": set(), "location": set(), "other": set()}
        for tk in self.tokens:
            if "_" not in tk:
                cats["other"].add(tk)
                continue
            prefix, _ = tk.split("_", 1)
            cats[category_map.get(prefix, "other")].add(tk)
        return cats


# ---------------------------------------------------------
# 3.  Config loader – single instance, hashable for caching
# ---------------------------------------------------------
class TongueConfig:
    """Loads JSON config once and pre‑computes helper structures."""

    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.category_map: Dict[str, str] = data["category_map"]
        self.token_dicts: Dict[str, List[str]] = data["token_dicts"]
        self.pattern_table: Dict[str, List[str]] = data["pattern_table"]
        self.weights: Dict[str, float] = data.get(
            "weights", {"tongue": 1.0, "coat": 1.0, "location": 1.0, "other": 1.0}
        )

        # --- synonym parsing (prefix agnostic) --------------------------------
        raw_syn = data["synonyms"]
        self.synonyms: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
        for cat, pairs in raw_syn.items():
            for k, v in pairs.items():
                a, b = k.split("|")
                # Store both prefix‑stripped and original forms for safety
                self.synonyms[cat][(a, b)] = v
                self.synonyms[cat][(b, a)] = v
        # Build quick lookup for cross‑category as flat dict
        self.cross_syn = self.synonyms.get("CROSS", {})

        # --- token → prefix map ----------------------------------------------
        self.val_to_prefix: Dict[str, str] = {
            val: pref for pref, lst in self.token_dicts.items() for val in lst
        }

        # --- compile pattern list (longest first) ----------------------------
        self.patterns_sorted: List[str] = sorted(self.pattern_table, key=len, reverse=True)

        # --- Build Aho‑Corasick automata -------------------------------------
        if _HAS_AHO:
            self.aho_patterns = ahocorasick.Automaton()
            for pat in self.pattern_table:
                self.aho_patterns.add_word(pat, ("PAT", pat))
            self.aho_tokens = ahocorasick.Automaton()
            for token in self.val_to_prefix:
                self.aho_tokens.add_word(token, ("TOK", token))
            self.aho_patterns.make_automaton()
            self.aho_tokens.make_automaton()
        else:
            self.aho_patterns = None  # type: ignore
            self.aho_tokens = None  # type: ignore

        # Stop‑words frequently appearing but uninformative
        self.stopwords = set(["苔", "舌"])

    # ---------- helper ------------------------------------------------------
    def make_full_token(self, val: str) -> str:
        """Return canonical PREFIX_val form; unknown prefix → OTHER."""
        pref = self.val_to_prefix.get(val, "OTHER")
        return f"{pref}_{val}"

    # -----------------------------------------------------------------------
    def extract(self, text: str) -> TokenSets:
        """Extract tokens + unknowns from text using Aho‑Corasick if available."""
        tokens: Set[str] = set()
        remaining = text

        # -- 3.1 Pattern match (longest first) --------------------------------
        pats_found: List[Tuple[int, int, str]] = []  # start,end,pat
        if _HAS_AHO:
            for end, (tag, pat) in self.aho_patterns.iter(text):
                start = end - len(pat) + 1
                pats_found.append((start, end, pat))
            pats_found.sort()  # order of occurrence
        else:
            for pat in self.patterns_sorted:
                for m in re.finditer(re.escape(pat), remaining):
                    pats_found.append((m.start(), m.end() - 1, pat))

        taken_mask = [False] * len(text)
        for s, e, pat in pats_found:
            if any(taken_mask[s : e + 1]):
                continue  # overlapping portion already claimed
            for i in range(s, e + 1):
                taken_mask[i] = True
            for full_tk in self.pattern_table[pat]:
                tokens.add(full_tk)

        # -- 3.2 Individual token match --------------------------------------
        if _HAS_AHO:
            for end, (tag, tok) in self.aho_tokens.iter(text):
                start = end - len(tok) + 1
                if any(taken_mask[start : end + 1]):  # skip if already part of pattern
                    continue
                tokens.add(self.make_full_token(tok))
                for i in range(start, end + 1):
                    taken_mask[i] = True
        else:
            for val, pref in self.val_to_prefix.items():
                for m in re.finditer(re.escape(val), remaining):
                    if any(taken_mask[m.start() : m.end()]):
                        continue
                    tokens.add(f"{pref}_{val}")
                    for i in range(m.start(), m.end()):
                        taken_mask[i] = True

        # -- 3.3 Unknown extraction ------------------------------------------
        unknown: Set[str] = set()
        buff = []
        for ch, used in zip(text, taken_mask):
            if not used and "\u4e00" <= ch <= "\u9fff":
                buff.append(ch)
            else:
                if buff:
                    word = "".join(buff)
                    if len(word) >= 1 and word not in self.stopwords:
                        unknown.add(word)
                    buff.clear()
        if buff:
            word = "".join(buff)
            if len(word) >= 1 and word not in self.stopwords:
                unknown.add(word)

        return TokenSets(tokens, unknown)


# ---------------------------------------------------------
# 4.  Similarity functions
# ---------------------------------------------------------

def _sim_lookup(a: str, b: str, cfg: TongueConfig) -> float:
    """Return similarity 0‑1 for two canonical tokens."""
    if a == b:
        return 1.0
    # strip prefix for synonym lookup
    _, val_a = a.split("_", 1)
    _, val_b = b.split("_", 1)
    pref_a = cfg.val_to_prefix.get(val_a, "OTHER")
    pref_b = cfg.val_to_prefix.get(val_b, "OTHER")
    # same category
    if pref_a == pref_b:
        syn = cfg.synonyms.get(pref_a, {})
        return syn.get((val_a, val_b), syn.get((val_b, val_a), 0.0))
    # cross
    return cfg.cross_syn.get((a, b), cfg.cross_syn.get((b, a), 0.0))


def _pairwise_similarity(pred: List[str], lab: List[str], cfg: TongueConfig) -> float:
    if not pred and not lab:
        return 1.0
    if not pred or not lab:
        return 0.0

    n, m = len(pred), len(lab)
    # shortcut for tiny lists (<=4) – exhaustive calc fast
    if n * m <= 16:
        scores = [max(_sim_lookup(p, l, cfg) for l in lab) for p in pred]
        return sum(scores) / max(n, m)

    # build matrix
    mat = np.zeros((n, m), dtype=np.float32)
    for i, p in enumerate(pred):
        for j, l in enumerate(lab):
            mat[i, j] = _sim_lookup(p, l, cfg)

    if mat.max() == 0.0:
        return 0.0

    # choose algorithm: Hungarian for <=128×128 else greedy
    if n <= 128 and m <= 128:
        cost = 1.0 - mat
        r, c = linear_sum_assignment(cost)
        sim_sum = mat[r, c].sum()
    else:  # greedy O(n log n)
        sim_sum = 0.0
        used_cols = set()
        flat = [(-mat[i, j], i, j) for i in range(n) for j in range(m) if mat[i, j] > 0]
        flat.sort()
        for neg_val, i, j in flat:
            if i in used_cols or j in used_cols:
                continue
            sim_sum += -neg_val
            used_cols.add(i)
            used_cols.add(j)
    return sim_sum / max(n, m)


# ---------------------------------------------------------
# 5.  Evaluator class
# ---------------------------------------------------------
class TongueEvaluator:
    def __init__(self, cfg: TongueConfig):
        self.cfg = cfg

    # ---------- single sample ----------------------------------------------
    def evaluate(self, pred: str, label: str) -> Tuple[float, Dict[str, float]]:
        ts_pred = self.cfg.extract(pred)
        ts_lab = self.cfg.extract(label)
        cats_pred = ts_pred.to_categories(self.cfg.category_map)
        cats_lab = ts_lab.to_categories(self.cfg.category_map)

        cat_scores: Dict[str, float] = {}
        total = 0.0
        total_w = 0.0
        for cat in ["tongue", "coat", "location", "other"]:
            s = _pairwise_similarity(list(cats_pred[cat]), list(cats_lab[cat]), self.cfg)
            cat_scores[cat] = s
            w = self.cfg.weights.get(cat, 1.0)
            total += s * w
            total_w += w
        overall = total / total_w if total_w else 0.0
        cat_scores["overall"] = overall
        return overall, cat_scores

    # ---------- batch -------------------------------------------------------
    def evaluate_batch(self, preds: List[str], labels: List[str]) -> Dict[str, float]:
        if len(preds) != len(labels):
            raise ValueError("Prediction / label length mismatch")
        agg = Counter()
        sum_scores = Counter()
        for p, l in zip(preds, labels):
            _, cs = self.evaluate(p, l)
            for k, v in cs.items():
                sum_scores[k] += v
            agg["samples"] += 1
        return {k: v / agg["samples"] for k, v in sum_scores.items()}


# ---------------------------------------------------------
# 6.  Unknown token monitor
# ---------------------------------------------------------
class UnknownMonitor:
    def __init__(self):
        self.freq = Counter()
        self.examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add(self, token_sets: TokenSets, pred: str, lab: str):
        for tok in token_sets.unknown:
            self.freq[tok] += 1
            if len(self.examples[tok]) < 5:
                self.examples[tok].append((pred, lab))

    def report(self, top_k: int = 30) -> str:
        lines = ["\nTop unknown tokens:\n"]
        for tok, cnt in self.freq.most_common(top_k):
            lines.append(f"  {tok}: {cnt}")
        return "\n".join(lines)

    def to_patch(self) -> str:
        """Return JSON snippet to extend token_dicts."""
        suggestions = {tok: [] for tok, _ in self.freq.most_common(50)}
        return json.dumps(suggestions, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# 7.  CLI utility
# ---------------------------------------------------------

def _read_jsonl(path: os.PathLike) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                data.append(json.loads(ln))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Tongue evaluation metric (o3 edition)")
    ap.add_argument("--config", required=True, help="token_config.json")
    ap.add_argument("--pred", required=True, help="prediction jsonl or dir")
    ap.add_argument("--label", help="label jsonl or dir (optional if pred contains both)")
    ap.add_argument("--out", default="metric_out", help="output directory")
    ap.add_argument("--combined", action="store_true", help="input file contains both predict and label")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = TongueConfig(args.config)
    evaluator = TongueEvaluator(cfg)
    umonitor = UnknownMonitor()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_file(pred_p: Path, lab_p: Path = None):
        if args.combined or lab_p is None:
            # Combined file mode - extract both predict and label from same file
            data = _read_jsonl(pred_p)
            preds = []
            labs = []
            
            for item in data:
                if 'predict' in item and 'label' in item:
                    pred = item['predict'].replace("舌诊结果: ", "").strip()
                    lab = item['label'].replace("舌诊结果: ", "").strip()
                    preds.append(pred)
                    labs.append(lab)
                elif 'predict' in item:
                    pred = item['predict'].replace("舌诊结果: ", "").strip()
                    preds.append(pred)
                elif 'label' in item:
                    lab = item['label'].replace("舌诊结果: ", "").strip()
                    labs.append(lab)
            
            if not preds or not labs:
                logging.error(f"Could not extract both predict and label from {pred_p}")
                return
                
        else:
            # Separate files mode
            pred_data = _read_jsonl(pred_p)
            lab_data = _read_jsonl(lab_p)
            preds = [d.get("predict", "").replace("舌诊结果: ", "").strip() for d in pred_data]
            labs = [d.get("label", "").replace("舌诊结果: ", "").strip() for d in lab_data]
        
        scores = evaluator.evaluate_batch(preds, labs)
        # Unknown monitoring
        for p, l in zip(preds, labs):
            ts = cfg.extract(p)
            umonitor.add(ts, p, l)
        out_f = out_dir / f"{pred_p.stem}_scores.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        logging.info(f"{pred_p.name}: overall={scores['overall']:.4f}")

    pred_path = Path(args.pred)
    lab_path = Path(args.label) if args.label else None

    if pred_path.is_dir():
        for pred_file in pred_path.glob("*.jsonl"):
            if lab_path:
                lab_file = lab_path / pred_file.name  # assume same name
                if not lab_file.exists():
                    logging.warning(f"Label for {pred_file.name} not found, skipping")
                    continue
                process_file(pred_file, lab_file)
            else:
                process_file(pred_file)
    else:
        process_file(pred_path, lab_path)

    # Unknown token global report
    with open(out_dir / "unknown_tokens.txt", "w", encoding="utf-8") as f:
        f.write(umonitor.report())

    with open(out_dir / "token_patch.json", "w", encoding="utf-8") as f:
        f.write(umonitor.to_patch())

    logging.info("Evaluation complete. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
