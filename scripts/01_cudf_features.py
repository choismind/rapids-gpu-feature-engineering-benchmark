"""기사 1번 경로: cuDF / cudf.pandas 로 만드는 피처들.

기사는 노트북 기준으로 `%load_ext cudf.pandas` 를 쓰라고 했는데,
그건 IPython 매직이라 .py 스크립트에서는 그대로 쓸 수 없다. 스크립트에서는
    python -m cudf.pandas 01_cudf_features.py
로 실행하거나, pandas import 전에 cudf.pandas.install() 을 호출해야 한다.
이 파일은 문법을 바꾸지 않는다는 기사의 주장을 그대로 검증하기 위해
pandas 코드만 담고, 백엔드 전환은 실행 방법으로만 처리한다.
"""

import pandas as pd

from common import active_backend, banner, demo_frame

banner(f"백엔드: {active_backend()}")

df = demo_frame()
print(df.to_string(index=False))

banner("① groupby 집계 — 브랜드별 평균 가격")
df["brand_avg_price"] = df.groupby("brand")["price"].transform("mean")
print(df[["brand", "price", "brand_avg_price"]].to_string(index=False))

banner("② 구간화 — price 를 low/medium/high 로")
df["price_bin"] = pd.cut(
    df["price"],
    bins=[0, 100, 150, float("inf")],
    labels=["low", "medium", "high"],
)
print(df[["price", "price_bin"]].to_string(index=False))

banner("③ 컬럼 결합 — brand × category 상호작용 피처")
df["brand_category"] = df["brand"].astype(str) + "_" + df["category"].astype(str)
print(df[["brand", "category", "brand_category"]].to_string(index=False))

banner("결과 검증")
avg_acme = df.loc[df["brand"] == "Acme", "brand_avg_price"].iloc[0]
print(f"Acme 평균 가격 = {avg_acme:.2f}  (기대: (80+120+250+45+130)/5 = 125.00)")
assert abs(float(avg_acme) - 125.0) < 1e-6, "groupby 결과가 기대와 다르다"
print(f"brand_category 고유값 = {df['brand_category'].nunique()}개")
print("OK — 세 연산 모두 같은 pandas 문법으로 동작했다.")
