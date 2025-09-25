import argparse

#from __future__ import annotations
import sys
import math
from itertools import combinations
from typing import List, Set, Tuple, Dict, Iterable

Item = str
Transaction = Set[Item]
Itemset = Tuple[Item, ...]

def find_f1_itemsets(T: List[Transaction], min_cnt: int) -> Dict[Itemset, int]:
    counts: Dict[Item, int] = {}
    for t in T:
        for it in t:
            counts[it] = counts.get(it, 0) + 1

    L1: Dict[Itemset, int] = {}
    for it, c in counts.items():
        if c >= min_cnt:
            L1[(it,)] = c
    return L1

def aprioriGeneration(Lk: Dict[Itemset, int], k: int) -> Dict[Itemset, int]:
    candidates: Dict[Itemset, int] = {}

    keys = sorted(Lk.keys())
    for a in keys:
        for b in keys:
            if a[:k - 1] == b[:k - 1] and a[k - 1] < b[k - 1]:
                c = a + (b[k - 1],)
                # Prune: all k-subsets of c must be in Lk
                if notPruneIf_has_all_k_subsets_in_Lk(c, Lk):
                    candidates[c] = 0
    return candidates


def notPruneIf_has_all_k_subsets_in_Lk(c: Itemset, Lk: Dict[Itemset, int]) -> bool:
    for idx in range(len(c)):
        subset = c[:idx] + c[idx + 1:]
        if subset not in Lk:
            return False
    return True


def count_supports(
    T: List[Transaction],
    candidates: Dict[Itemset, int]
) -> Dict[Itemset, int]:

    if not candidates:
        return candidates

    k = len(next(iter(candidates.keys())))
    cand_keys = set(candidates.keys())

    for t in T:
        if len(t) < k:
            continue
        sorted_t = tuple(sorted(t))
        for s in combinations(sorted_t, k):
            if s in cand_keys:
                candidates[s] += 1
    return candidates

def apriori_algorithm(
    T: List[Transaction],
    minsup_percent: float
) -> List[Tuple[Itemset, int]]:
    """
    Full Apriori pipeline. Returns list of (itemset, support_count) across all levels.
    """
    min_cnt = int(math.ceil((minsup_percent * n_tx) / 100.0))

    # L1: frequent 1-itemsets
    Lk = find_f1_itemsets(T, min_cnt)
    all_frequents: List[Tuple[Itemset, int]] = sorted(Lk.items())
    k = 1
    while Lk:
        Ckplus1 = aprioriGeneration(Lk, k)  # candidates of size k+1
        Ckplus1 = count_supports(T, Ckplus1)
        LkPlus1 = {iset: cnt for iset, cnt in Ckplus1.items() if cnt >= min_cnt}

        all_frequents.extend(LkPlus1.items())
        Lk = LkPlus1
        k += 1
    return all_frequents

def read_dataset(filename: str) -> List[Transaction]:
    """
    Read transactions from file.
    Each line is a transaction; items are separated by spaces.
    Duplicate items within a transaction are removed.
    Empty lines are ignored.
    """
    T: List[Transaction] = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split()
            # Deduplicate within a transaction; keep as set
            T.append(set(items))
    return T

parser = argparse.ArgumentParser(description="Run Apriori algorithm with minsup.")
parser.add_argument("filename", type=str, nargs="?", help="Dataset file", default="dataset1.txt")
parser.add_argument("minsup", type=int, nargs="?", help="Minimum support value", default=10)
args = parser.parse_args()

filename = args.filename
minsup_percent = args.minsup
T = read_dataset(filename)

n_tx = len(T)
if n_tx == 0:
    print("No transactions found in the input file.")
else:
    results = apriori_algorithm(T, minsup_percent)
    if not results:
        print("No frequent itemsets (given the specified minimum support).")
    else:
        print(f"total frequent itemset:{len(results)}\nItemset : support (Percent)")
        for itemset, cnt in results:
            pct = (100.0 * cnt) / n_tx if n_tx > 0 else 0.0
            left = " ".join(itemset)
            print(f"{left} : {cnt} ({pct:.2f}%)")