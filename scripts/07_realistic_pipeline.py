"""현실적인 파이프라인 한 판을 통째로 재 본다.

기사가 인용한 캐글 사례의 형태를 그대로 만든다:
  범주형 8컬럼 -> 모든 쌍 조합 28개 -> 각 조합마다 5-fold out-of-fold 타깃 인코딩

연산 단위 마이크로벤치(05번)와 달리, 이건 "피처 한 배치를 만드는 데 걸리는 총 시간"이다.
피처 아이디어를 몇 번 돌려볼 수 있느냐를 결정하는 숫자다.

    python                07_realistic_pipeline.py   # CPU
    python -m cudf.pandas 07_realistic_pipeline.py   # GPU
"""

import itertools
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd

from common import active_backend, banner, is_gpu_pandas

N_ROWS = int(os.environ.get("N_ROWS", 10_000_000))
N_CAT_COLS = int(os.environ.get("N_CAT_COLS", 8))
N_FOLDS = 5

banner(f"백엔드: {active_backend()}")
print(f"행 {N_ROWS:,} / 범주형 컬럼 {N_CAT_COLS}개 / 폴드 {N_FOLDS}")

# ---------------------------------------------------------------- 데이터 생성
rng = np.random.default_rng(0)
cardinalities = [8, 6, 12, 4, 20, 50, 100, 5][:N_CAT_COLS]

data = {}
for i, card in enumerate(cardinalities):
    vals = np.array([f"c{i}_v{j:03d}" for j in range(card)])
    data[f"cat{i}"] = pd.Categorical(vals[rng.integers(0, card, N_ROWS)])

signal = sum(rng.random(N_ROWS) * 0.1 for _ in range(3)) + 0.3
data["label"] = (rng.random(N_ROWS) < np.clip(signal, 0, 1)).astype("int8")

t0 = time.perf_counter()
df = pd.DataFrame(data)
print(f"데이터 프레임 구성: {time.perf_counter() - t0:.2f}s (측정 제외)")

cat_cols = [c for c in df.columns if c.startswith("cat")]
pairs = list(itertools.combinations(cat_cols, 2))
print(f"조합 쌍 {len(pairs)}개 (= C({len(cat_cols)},2))")

prior = float(df["label"].mean())
fold = pd.Series(np.arange(N_ROWS) % N_FOLDS, index=df.index)


def oof_target_encode(keys):
    """out-of-fold 타깃 인코딩 — 폴드 수만큼 groupby 를 반복한다."""
    tmp = pd.DataFrame({"k": keys, "y": df["label"], "f": fold})
    out = pd.Series(np.full(len(tmp), np.nan), index=tmp.index)
    for f in range(N_FOLDS):
        hold = tmp["f"] == f
        means = tmp.loc[~hold].groupby("k")["y"].mean()
        out[hold] = tmp.loc[hold, "k"].map(means).fillna(prior)
    return out


# ---------------------------------------------------------------- 파이프라인
banner("피처 배치 생성")
t_combine = 0.0
t_encode = 0.0
made = 0

t_all = time.perf_counter()
for a, b in pairs:
    t = time.perf_counter()
    combo = df[a].astype(str) + "_" + df[b].astype(str)
    _ = combo.iloc[0]                      # 지연 평가 방지
    t_combine += time.perf_counter() - t

    t = time.perf_counter()
    enc = oof_target_encode(combo)
    _ = enc.iloc[0]
    t_encode += time.perf_counter() - t

    made += 1
    del combo, enc                          # 메모리 회수 (결과는 버린다 — 시간만 잰다)

total = time.perf_counter() - t_all

banner("결과")
print(f"  생성한 피처            {made}개")
print(f"  ① 컬럼 결합 합계       {t_combine:8.2f} s")
print(f"  ② OOF 타깃 인코딩 합계 {t_encode:8.2f} s")
print(f"  ─────────────────────────────────")
print(f"  전체                   {total:8.2f} s")
print(f"  피처 1개당             {total / made:8.3f} s")

backend = "gpu" if is_gpu_pandas() else "cpu"
out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
path = out_dir / f"pipeline_{backend}_{N_ROWS}.json"
path.write_text(
    json.dumps(
        {
            "backend": backend,
            "n_rows": N_ROWS,
            "n_features": made,
            "combine_s": t_combine,
            "encode_s": t_encode,
            "total_s": total,
            "per_feature_s": total / made,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\n저장: {path}")
