"""Walk-forward study of NEW signal filters / scoring systems for the MA and MarketTrend strategies.

Candidate trades = every signal of the APK 0.2.1 base rules (slippage 0.05%, stop-first, per-frame wallets) on the
expanded universe (45 coins) and the longer history.  A filter only decides WHICH of those trades to take.
For every fold (test period) the filter is FITTED on trades that were already CLOSED before the test period started
and then applied unchanged to the test trades.  Pooled out-of-sample statistics evaluate the whole selection
PROCEDURE (not one hand-picked rule), so the numbers are not inflated by the search.
numpy only (no scikit-learn in this environment).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import apk_features as F
import apk_strategies as S

ALL_COINS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT", "TRX", "XLM", "ATOM", "ETC", "BCH", "NEAR",
             "AAVE", "UNI", "SUI", "APT", "ARB", "OP", "HBAR", "POL", "SC", "SHIB", "STX", "FIL", "SAND", "ALGO", "AXS",
             "WAVES", "KAVA", "VET", "ELF", "EGLD", "LSK", "ARK", "GAS", "CHZ", "BAT", "NEO", "IOTA", "IMX", "GRT", "ANKR"]
SCORE_COINS = ALL_COINS[:23]        # the market score keeps the original 23-coin definition
FOLDS = {
    "day": [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2021, 2027)],
    "60m": [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2022, 2027)],
    "30m": [(f"{y}-01-01", f"{y + 1}-01-01") for y in range(2024, 2027)],
    "5m": [("2026-03-01", "2026-05-01"), ("2026-05-01", "2026-07-01"), ("2026-07-01", "2026-09-01"), ("2026-09-01", "2026-12-01")],
}
QS = np.linspace(0.1, 0.9, 9)


def candidates(strategy: str, tf: str, sc, feats: dict) -> pd.DataFrame:
    S.CFG.update(ma_f={}, trend_f={})
    S.SLIP, S.ORDER_MODE = 0.0005, "stop_first"
    rows, fr = [], []
    for coin in ALL_COINS:
        b = S.load_bars(coin, tf)
        if b is None or coin not in feats:
            continue
        tr = S.ma_trades(coin, tf, b, sc) if strategy == "ma" else S.trend_trades(coin, tf, b, sc)
        if not tr:
            continue
        idx = np.array([t.sig_i for t in tr])
        f = feats[coin].iloc[idx].reset_index(drop=True)
        d = pd.DataFrame({"coin": coin, "entry_t": [t.entry_t for t in tr], "exit_t": [t.exit_t for t in tr],
                          "ret": [t.ret for t in tr], "size_factor": [t.size_factor for t in tr],
                          "reason": [t.reasons[-1] for t in tr]})
        rows.append(pd.concat([d, f], axis=1))
    T = pd.concat(rows, ignore_index=True).sort_values("entry_t").reset_index(drop=True)
    return T


def feature_cols(T: pd.DataFrame) -> list[str]:
    skip = {"coin", "entry_t", "exit_t", "ret", "size_factor", "reason"}
    return [c for c in T.columns if c not in skip and T[c].notna().mean() > 0.6]


# ----------------------------------------------------------------------------------- selection procedures
def tstat(r: np.ndarray) -> float:
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 and r.std(ddof=1) > 0 else -9.0


def _stable(r_sorted: np.ndarray) -> bool:
    h = len(r_sorted) // 2
    return h >= 8 and r_sorted[:h].mean() > 0 and r_sorted[h:].mean() > 0


def best_rule(tr: pd.DataFrame, cols: list[str], min_n: int, base_mask=None):
    """Single threshold rule (feature >= / <= train quantile) with the best t-stat and positive mean in both halves."""
    best = None
    ret = tr["ret"].to_numpy()
    bm = np.ones(len(tr), bool) if base_mask is None else base_mask
    for c in cols:
        x = tr[c].to_numpy(float)
        ok = ~np.isnan(x) & bm
        if ok.sum() < min_n:
            continue
        for q in QS:
            th = np.nanquantile(x[ok], q)
            for sgn in (1, -1):
                m = ok & ((x >= th) if sgn > 0 else (x <= th))
                n = m.sum()
                if n < min_n:
                    continue
                r = ret[m]
                if r.mean() <= 0 or not _stable(r):
                    continue
                t = tstat(r)
                if best is None or t > best[0]:
                    best = (t, c, sgn, float(th), n, float(r.mean()))
    return best


def apply_rule(df: pd.DataFrame, rule) -> np.ndarray:
    _, c, sgn, th, *_ = rule
    x = df[c].to_numpy(float)
    return ~np.isnan(x) & ((x >= th) if sgn > 0 else (x <= th))


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float = 5.0, it: int = 25) -> np.ndarray:
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(d + 1)
    R = lam * np.eye(d + 1); R[0, 0] = 0
    for _ in range(it):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R + 1e-6 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 50.0) -> np.ndarray:
    n, d = X.shape
    Xb = np.c_[np.ones(n), X]
    R = lam * np.eye(d + 1); R[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ y)


class Prep:
    """Robust standardisation fitted on train only; NaN -> 0 after scaling."""
    def __init__(self, tr: pd.DataFrame, cols: list[str]):
        self.cols = cols
        x = tr[cols].to_numpy(float)
        self.med = np.nanmedian(x, axis=0)
        iqr = np.nanquantile(x, 0.75, axis=0) - np.nanquantile(x, 0.25, axis=0)
        self.sc = np.where(iqr > 1e-12, iqr, 1.0)
        self.lo = np.nanquantile(x, 0.01, axis=0); self.hi = np.nanquantile(x, 0.99, axis=0)

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        x = np.clip(df[self.cols].to_numpy(float), self.lo, self.hi)
        z = (x - self.med) / self.sc
        return np.nan_to_num(z, nan=0.0)


def top_q_mask(score_tr: np.ndarray, score_te: np.ndarray, q: float) -> np.ndarray:
    return score_te >= np.quantile(score_tr, 1 - q)


def spearman_corr(tr: pd.DataFrame, c: str) -> float:
    x = tr[c].rank(pct=True)
    return float(x.corr(tr["ret"].rank(pct=True)))


def methods_for_fold(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], strategy: str, rng) -> dict[str, np.ndarray]:
    out = {"base": np.ones(len(te), bool)}
    if strategy == "ma":
        out["legacy: MA120 이격>=3%"] = te["d_ma120"].to_numpy(float) <= -0.03
    out["legacy: 시장점수>0"] = te["score"].to_numpy(float) > 0
    min_n = max(30, int(0.08 * len(tr)))
    r1 = best_rule(tr, cols, min_n)
    out["rule1 (단일 지표)"] = apply_rule(te, r1) if r1 else np.zeros(len(te), bool)
    if r1:
        m1 = apply_rule(tr, r1)
        r2 = best_rule(tr, [c for c in cols if c != r1[1]], max(25, int(0.04 * len(tr))), base_mask=m1)
        out["rule2 (2지표 AND)"] = (apply_rule(te, r1) & apply_rule(te, r2)) if r2 else out["rule1 (단일 지표)"]
        out["_r1"], out["_r2"] = r1, r2
    else:
        out["rule2 (2지표 AND)"] = np.zeros(len(te), bool)
    P = Prep(tr, cols)
    Xtr, Xte = P(tr), P(te)
    y = (tr["ret"].to_numpy() > 0).astype(float)
    w = fit_logit(Xtr, y)
    ptr, pte = np.c_[np.ones(len(Xtr)), Xtr] @ w, np.c_[np.ones(len(Xte)), Xte] @ w
    wr = fit_ridge(Xtr, np.clip(tr["ret"].to_numpy(), -0.15, 0.15))
    rtr, rte = np.c_[np.ones(len(Xtr)), Xtr] @ wr, np.c_[np.ones(len(Xte)), Xte] @ wr
    # composite rank score: top-6 features by |Spearman| with the net return (sign from train)
    cors = sorted(((abs(spearman_corr(tr, c)), np.sign(spearman_corr(tr, c)), c) for c in cols if tr[c].notna().mean() > 0.6),
                  reverse=True)[:6]
    def comp(df):
        s = np.zeros(len(df))
        for _, sg, c in cors:
            ref = np.sort(tr[c].dropna().to_numpy(float))
            pr = np.searchsorted(ref, df[c].to_numpy(float)) / max(1, len(ref))
            s += sg * np.nan_to_num(pr, nan=0.5)
        return s / len(cors)
    ctr, cte = comp(tr), comp(te)
    for q in (0.5, 0.3, 0.15):
        out[f"logit 점수 상위{int(q * 100)}%"] = top_q_mask(ptr, pte, q)
        out[f"ridge 점수 상위{int(q * 100)}%"] = top_q_mask(rtr, rte, q)
        out[f"순위합성 점수 상위{int(q * 100)}%"] = top_q_mask(ctr, cte, q)
        out[f"무작위 {int(q * 100)}% (잡음 기준)"] = rng.random(len(te)) < q
    out["_cors"] = [c for _, _, c in cors]
    return out


def evaluate(selected: pd.DataFrame, tf: str) -> dict:
    if len(selected) == 0:
        return dict(n=0)
    r = selected["ret"].to_numpy()
    objs = [SimpleNamespace(entry_t=a, exit_t=b, ret=c, size_factor=d)
            for a, b, c, d in zip(selected["entry_t"], selected["exit_t"], selected["ret"], selected["size_factor"])]
    w, l = r[r > 0], r[r <= 0]
    return dict(n=len(r), win=float((r > 0).mean()), avg=float(r.mean()), t=tstat(r),
                pf=float(w.sum() / -l.sum()) if l.sum() < 0 else np.nan, port=S.portfolio(objs, tf)["portfolio_return"])


def run(strategy: str, tf: str, T: pd.DataFrame, seed: int = 7):
    cols = feature_cols(T)
    rng = np.random.default_rng(seed)
    pooled: dict[str, list] = {}
    fold_rows, chosen = [], []
    for a, bnd in FOLDS[tf]:
        a, bnd = pd.Timestamp(a), pd.Timestamp(bnd)
        tr = T[T["exit_t"] < a]
        te = T[(T["entry_t"] >= a) & (T["entry_t"] < bnd)]
        if len(tr) < 150 or len(te) < 30:
            continue
        m = methods_for_fold(tr, te, cols, strategy, rng)
        meta = {k: m.pop(k) for k in [k for k in m if k.startswith("_")]}
        if "_r1" in meta:
            chosen.append(dict(strategy=strategy, tf=tf, fold=str(a.date()), r1=str(meta["_r1"][1:4]) if meta["_r1"] else "",
                               r2=str(meta["_r2"][1:4]) if meta.get("_r2") else "", top_corr=",".join(meta.get("_cors", []))))
        for name, mask in m.items():
            sel = te[mask]
            pooled.setdefault(name, []).append(sel)
            e = evaluate(sel, tf)
            fold_rows.append(dict(strategy=strategy, tf=tf, fold=str(a.date()), method=name, **e))
    rows = []
    for name, parts in pooled.items():
        sel = pd.concat(parts)
        e = evaluate(sel.sort_values("entry_t"), tf)
        row = dict(strategy=strategy, tf=tf, method=name, **e)
        if len(parts) >= 2:                                   # clean final test: last fold never used to pick a method
            pre = evaluate(pd.concat(parts[:-1]).sort_values("entry_t"), tf)
            last = evaluate(parts[-1].sort_values("entry_t"), tf)
            row.update({f"pre_{k}": v for k, v in pre.items()}); row.update({f"last_{k}": v for k, v in last.items()})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.DataFrame(chosen), cols


def main() -> None:
    tfs = sys.argv[1:] or ["day", "60m", "30m", "5m"]
    from apk_run import COINS
    ends, score = S.build_market_score(SCORE_COINS)
    sc = S.ScoreLookup(ends, score)
    res, folds, chosen, tabs = [], [], [], {}
    for tf in tfs:
        feats = F.build_all(ALL_COINS, tf, sc)
        for strategy in ("ma", "trend"):
            if strategy == "trend":
                S.TREND_WATCH_H["day"] = 240        # USER (2026-11-21): daily watch 10 days
            T = candidates(strategy, tf, sc, feats)
            T.to_pickle(f"/tmp/claude-0/cand_{strategy}_{tf}.pkl")
            print(f"[{strategy} {tf}] candidate trades {len(T)}  {T.entry_t.min().date()} -> {T.entry_t.max().date()}", flush=True)
            a, b, c, cols = run(strategy, tf, T)
            res.append(a); folds.append(b); chosen.append(c)
            print(a.round(3).to_string(index=False), flush=True)
    pd.concat(res).to_csv(S.RESULTS / "apk_wf_pooled.csv", index=False)
    pd.concat(folds).to_csv(S.RESULTS / "apk_wf_folds.csv", index=False)
    pd.concat(chosen).to_csv(S.RESULTS / "apk_wf_chosen_rules.csv", index=False)


if __name__ == "__main__":
    main()
