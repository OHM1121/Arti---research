"""Walk-forward check of the grid search: pick a combo using only past data, then test it on the next window.

Folds (selection uses data up to the fold's cut-off only):
  Fold 1: select on 2020-09 ~ 2022-12, test on 2023-01 ~ 2024-12
  Fold 2: select on 2020-09 ~ 2024-12, test on 2025-01 ~ 2026-10

Selection rule (fixed before looking at test data): among combos with >= 30 trades in the selection
window, pick the highest profit factor. Test window PF is only reported after the pick.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

import backtest as bt
from search import GRID, pf_of

MIN_TRADES = 30
FOLDS = [
    ("2022-12-31", "2023-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2026-10-31"),
]


def main() -> None:
    p = bt.load_panel()
    ind = bt.add_indicators(p)
    keys = list(GRID)
    runs = []
    for vals in itertools.product(*[GRID[k] for k in keys]):
        params = dict(zip(keys, vals))
        _, _, tr, _ = bt.simulate(p, ind, replace(bt.Config(), **params))
        runs.append((params, tr))
    print(f"simulated {len(runs)} combos")

    out_rows = []
    for sel_end, test_start, test_end in FOLDS:
        sel_end, test_start, test_end = map(pd.Timestamp, (sel_end, test_start, test_end))
        scored = []
        for i, (params, tr) in enumerate(runs):
            pf, n, _ = pf_of(tr, end=sel_end)
            scored.append((i, params, pf, n))
        eligible = [s for s in scored if s[3] >= MIN_TRADES and np.isfinite(s[2])]
        eligible.sort(key=lambda s: -s[2])
        best_i, best_params, sel_pf, sel_n = eligible[0]
        test_pf, test_n, test_win = pf_of(runs[best_i][1], start=test_start, end=test_end)
        # distribution of test PF for ALL eligible combos, to see whether selection helps at all
        all_test = []
        for prm, tr in runs:
            pf, n, _ = pf_of(tr, end=sel_end)
            if n >= MIN_TRADES and np.isfinite(pf):
                tpf, tn, _ = pf_of(tr, start=test_start, end=test_end)
                if tn >= 5 and np.isfinite(tpf):
                    all_test.append(tpf)
        all_test = pd.Series(all_test)
        out_rows.append({
            "select_until": sel_end.date(), "test": f"{test_start.date()}~{test_end.date()}",
            "picked": {k: best_params[k] for k in keys},
            "select_PF": round(sel_pf, 3), "select_trades": sel_n,
            "test_PF": round(test_pf, 3), "test_trades": test_n, "test_win": round(test_win, 3),
            "test_PF_median_all_eligible": round(all_test.median(), 3),
            "share_test_PF>=1.01_all_eligible": round((all_test >= 1.01).mean(), 3),
        })

    for r in out_rows:
        print("\n" + "=" * 90)
        print(f"select on data up to {r['select_until']} -> test {r['test']}")
        print("picked:", r["picked"])
        print(f"  select PF {r['select_PF']} ({r['select_trades']} trades)")
        print(f"  TEST   PF {r['test_PF']} ({r['test_trades']} trades, win {r['test_win']})")
        print(f"  for comparison, median test PF of all eligible combos: {r['test_PF_median_all_eligible']}")
        print(f"  share of eligible combos with test PF>=1.01: {r['share_test_PF>=1.01_all_eligible']}")
    pd.DataFrame(out_rows).to_csv(bt.RESULT_DIR / "walkforward.csv", index=False)


if __name__ == "__main__":
    main()
