"""규모를 키웠을 때 CPU vs GPU 차이를 실측한다 (기사에는 수치가 없다).

같은 코드를 두 번 돌린다:
    python     05_benchmark_features.py     -> CPU
    python -m cudf.pandas 05_benchmark_features.py -> GPU
행 수는 N_ROWS 환경변수로 조절한다 (기본 1000만).
"""

import json
import os
import pathlib
import time

import pandas as pd

from common import active_backend, banner, big_frame, is_gpu_pandas

N_ROWS = int(os.environ.get("N_ROWS", 10_000_000))
N_PRODUCTS = int(os.environ.get("N_PRODUCTS", 100_000))

banner(f"백엔드: {active_backend()}")
print(f"행 수 {N_ROWS:,} / product_id 고유값 {N_PRODUCTS:,}")

t0 = time.perf_counter()
df = big_frame(N_ROWS, N_PRODUCTS)
print(f"데이터 생성(CPU numpy, 측정 제외): {time.perf_counter() - t0:.2f}s")


def timed(name, fn, repeat=3):
    """첫 회는 워밍업으로 버리고 최소값을 취한다. 결과를 스칼라로 만들어 실제 계산을 강제한다."""
    fn()  # warmup / JIT / 커널 로딩
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        _ = out.iloc[0] if hasattr(out, "iloc") else out  # 지연 평가 방지
        best = min(best, time.perf_counter() - t)
    print(f"  {name:<34} {best*1000:9.1f} ms")
    return best


banner("피처 생성 연산")
results = {}
results["groupby_transform_mean"] = timed(
    "① groupby brand -> mean price", lambda: df.groupby("brand")["price"].transform("mean")
)
results["groupby_highcard"] = timed(
    "② groupby product_id -> mean label", lambda: df.groupby("product_id")["label"].mean()
)
results["binning"] = timed(
    "③ pd.cut 구간화",
    lambda: pd.cut(df["price"], bins=[0, 100, 150, float("inf")],
                   labels=["low", "medium", "high"]),
)
results["string_combine"] = timed(
    "④ brand + '_' + category 결합",
    lambda: df["brand"].astype(str) + "_" + df["category"].astype(str),
)
results["value_counts"] = timed(
    "⑤ product_id value_counts", lambda: df["product_id"].value_counts()
)


def oof_target_encode(frame, col, n_folds=5):
    """기사가 말한 out-of-fold 타깃 인코딩을 손으로 구현 — 폴드 수만큼 groupby 가 반복된다."""
    import numpy as np

    fold = np.arange(len(frame)) % n_folds
    prior = float(frame["label"].mean())
    out = pd.Series(np.full(len(frame), np.nan), index=frame.index)
    for f in range(n_folds):
        holdout = fold == f
        rest = frame[~holdout]
        means = rest.groupby(col)["label"].mean()
        enc = frame.loc[holdout, col].map(means).fillna(prior)
        out[holdout] = enc
    return out


results["oof_te_5folds"] = timed(
    f"⑥ OOF 타깃 인코딩 product_id (5 folds)",
    lambda: oof_target_encode(df, "product_id"),
    repeat=1,
)

banner("요약")
backend = "gpu" if is_gpu_pandas() else "cpu"
for k, v in results.items():
    print(f"  {k:<26} {v*1000:9.1f} ms")

out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
path = out_dir / f"bench_{backend}_{N_ROWS}.json"
path.write_text(json.dumps({"backend": backend, "n_rows": N_ROWS,
                            "n_products": N_PRODUCTS, "seconds": results}, indent=2),
                encoding="utf-8")
print(f"\n저장: {path}")
