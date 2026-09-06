"""기사가 지적한 타깃 인코딩의 두 가지 함정을 손으로 재현한다.

  (1) 희소 범주    — Solo 는 1행뿐이라 평균이 1.00 이 된다
  (2) 타깃 누수    — 자기 행의 라벨이 자기 인코딩 값에 들어간다

그리고 각각의 처방(스무딩 / out-of-fold)이 숫자를 어떻게 바꾸는지 확인한다.
백엔드는 CPU/GPU 어느 쪽이든 같은 코드로 돈다.
"""

import numpy as np
import pandas as pd

from common import active_backend, banner, demo_frame

banner(f"백엔드: {active_backend()}")

df = demo_frame()

banner("① 순진한 타깃 인코딩 (기사 본문 그대로)")
brand_te = (
    df.groupby("brand", as_index=False)["label"]
    .mean()
    .rename(columns={"label": "brand_TE"})
)
counts = df.groupby("brand", as_index=False)["label"].size().rename(columns={"size": "n"})
naive = brand_te.merge(counts, on="brand")
print(naive.to_string(index=False))

acme = float(naive.loc[naive["brand"] == "Acme", "brand_TE"].iloc[0])
solo = float(naive.loc[naive["brand"] == "Solo", "brand_TE"].iloc[0])
print(f"\nAcme = {acme:.2f}  (기사 본문의 0.60 과 일치해야 한다)")
print(f"Solo = {solo:.2f}  <- 리뷰 1건짜리인데 100% 로 단정된다. 이게 문제 (1).")
assert abs(acme - 0.60) < 1e-9, "기사의 0.60 을 재현하지 못했다"
assert abs(solo - 1.00) < 1e-9

banner("② 스무딩 — 범주 평균과 전체 평균을 섞는다")
prior = float(df["label"].mean())
print(f"전체 평균(prior) = {prior:.3f}")
print("  smoothed = (n * category_mean + smooth * prior) / (n + smooth)\n")

rows = []
for smooth in (0, 1, 5, 20):
    s = (naive["n"] * naive["brand_TE"] + smooth * prior) / (naive["n"] + smooth)
    rows.append(pd.DataFrame({"smooth": smooth, "brand": naive["brand"], "TE": s.round(3)}))
wide = pd.concat(rows).pivot(index="brand", columns="smooth", values="TE")
print(wide.to_string())
print("\nsmooth 가 커질수록 Solo 가 전체 평균 쪽으로 끌려 내려간다 — 처방 (1) 확인.")

banner("③ 타깃 누수와 out-of-fold 인코딩")
print("Acme 첫 행(label=1)을 인코딩할 때, 순진한 방식은 그 행의 라벨을 이미 쓰고 있다.\n")

n_folds = 5
rng = np.random.default_rng(42)
fold = rng.integers(0, n_folds, len(df))
df["fold"] = fold

oof = np.full(len(df), np.nan)
for f in range(n_folds):
    holdout = df["fold"] == f
    rest = ~holdout
    if not rest.any():
        continue
    means = df.loc[rest].groupby("brand")["label"].mean()
    fallback = float(df.loc[rest, "label"].mean())
    oof[holdout.to_numpy()] = (
        df.loc[holdout, "brand"].map(means).fillna(fallback).to_numpy()
    )
df["brand_TE_oof"] = oof
df["brand_TE_naive"] = df["brand"].map(naive.set_index("brand")["brand_TE"])

print(df[["brand", "label", "fold", "brand_TE_naive", "brand_TE_oof"]].to_string(index=False))
print("\n같은 브랜드인데 fold 마다 값이 다르다 = 자기 라벨을 쓰지 않았다는 증거 — 처방 (2) 확인.")
print(f"계산 횟수: 순진한 방식 1회 vs OOF {n_folds}회.  피처가 F개면 F×{n_folds}회로 불어난다.")

banner("④ 조합 인코딩 — brand × category")
combo = (
    df.groupby(["brand", "category"], as_index=False)["label"]
    .mean()
    .rename(columns={"label": "brand_category_TE"})
)
print(combo.to_string(index=False))
print(f"\n단일 컬럼 2개 -> 조합 1개로 그룹 수가 {df['brand'].nunique()}+{df['category'].nunique()}"
      f" = {df['brand'].nunique() + df['category'].nunique()} 에서 {len(combo)} 로 늘었다.")
