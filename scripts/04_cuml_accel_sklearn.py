"""기사 3번 주장: cuml.accel 을 켜면 scikit-learn 코드를 한 줄도 안 고치고 GPU 로 보낸다.

기사 원문: "cuml.accel keeps the scikit-learn behavior".
이 스크립트는 그 주장을 곧이곧대로 믿지 않고 출력값으로 대조한다.

노트북의 `%load_ext cuml.accel` 은 IPython 매직이다. 스크립트에서는
    python -m cuml.accel 04_cuml_accel_sklearn.py
로 실행한다. 이 파일 자체에는 sklearn 코드만 있다.

    python                04_cuml_accel_sklearn.py   # 순정 sklearn (CPU)
    python -m cuml.accel  04_cuml_accel_sklearn.py   # 가속
두 출력을 나란히 비교할 것.
"""

import sys

import numpy as np
import pandas as pd

from common import banner, big_frame, demo_frame

banner("cuml.accel 적용 여부")
from sklearn.preprocessing import TargetEncoder

accel_loaded = "cuml.accel" in sys.modules
print("TargetEncoder ->", f"{TargetEncoder.__module__}.{TargetEncoder.__qualname__}")
print("cuml.accel 로딩:", accel_loaded)

df = demo_frame()
train = df.iloc[:8].reset_index(drop=True)
valid = df.iloc[8:].reset_index(drop=True)

# 주의: sklearn TargetEncoder 는 이진 타깃에 StratifiedKFold 를 쓴다.
# 8행(클래스당 4개)에 cv=5 를 주면 순정 sklearn 은 ValueError 로 죽는다.
CV = 3

banner(f"① sklearn TargetEncoder — 코드 변경 없음 (cv={CV})")
enc = TargetEncoder(categories="auto", smooth=20, cv=CV, shuffle=True, random_state=42)
tr = enc.fit_transform(train[["brand"]], train["label"])
va = enc.transform(valid[["brand"]])

out = train[["brand", "label"]].copy()
out["brand_TE"] = np.asarray(tr).ravel()
print(out.to_string(index=False))
print("\nvalid:", np.asarray(va).ravel())

banner("② 여러 컬럼을 넣으면? — 기사가 말한 차이 확인")
enc2 = TargetEncoder(categories="auto", smooth=20, cv=CV, shuffle=True, random_state=42)
arr = np.asarray(enc2.fit_transform(train[["brand", "category"]], train["label"]))
print("출력 shape:", arr.shape)
if arr.ndim == 2 and arr.shape[1] == 2:
    print("-> 컬럼 2개. brand 와 category 가 '각각 따로' 인코딩되었다.")
    print("   기사 지적대로 결합(combination) 인코딩이 아니다.")
else:
    print("-> 예상과 다른 shape. 실제 동작을 다시 확인할 것.")

banner("③ 핵심 검증 — fit_transform 이 정말 out-of-fold 인가?")
print("sklearn 의 fit_transform 은 폴드마다 다른 값을 내야 한다(타깃 누수 방지).")
print("같은 범주값의 인코딩이 '한 가지 값뿐'이라면 OOF 가 아니라 전체 데이터로 인코딩한 것이다.\n")

big = big_frame(n_rows=20_000, n_products=50, seed=7)
Xb = big[["brand"]].astype(str)
yb = big["label"].to_numpy()

enc3 = TargetEncoder(categories="auto", smooth=20, cv=5, shuffle=True, random_state=42)
enc_out = np.asarray(enc3.fit_transform(Xb, yb)).ravel()

chk = pd.DataFrame({"brand": Xb["brand"].to_numpy(), "te": enc_out})
distinct = chk.groupby("brand")["te"].nunique().sort_index()
print("브랜드별 서로 다른 인코딩 값의 개수 (cv=5 이므로 OOF 라면 5 근처여야 한다):")
print(distinct.to_string())

max_distinct = int(distinct.max())
if max_distinct == 1:
    print("\n>>> 값이 범주당 1개뿐 = out-of-fold 가 아니다. 타깃 누수가 그대로 남는다.")
    print("    (기사의 'keeps the scikit-learn behavior' 와 어긋나는 지점)")
else:
    print(f"\n>>> 최대 {max_distinct}개 = 폴드별로 값이 갈린다. out-of-fold 동작 확인.")

banner("④ 전체 데이터 인코딩 값과 대조")
prior = float(yb.mean())
grp = chk.assign(y=yb).groupby("brand")["y"]
full = ((grp.sum() + 20 * prior) / (grp.count() + 20)).sort_index()
first = chk.groupby("brand")["te"].first().sort_index()
cmp = pd.DataFrame({"fit_transform_첫값": first, "전체데이터_스무딩값": full})
cmp["차이"] = (cmp["fit_transform_첫값"] - cmp["전체데이터_스무딩값"]).abs()
print(cmp.to_string())
print(f"\n최대 차이 = {cmp['차이'].max():.6f}")
print("차이가 0 에 붙으면 fit_transform 이 전체 데이터를 그대로 쓴 것이다.")

banner("결론")
print("cuml.accel 은 '코드를 안 고쳐도 돈다'는 편의가 목적이지,")
print("sklearn 과 값이 같다는 보장은 아니다. 위 ③④ 결과를 CPU 실행과 반드시 대조할 것.")
