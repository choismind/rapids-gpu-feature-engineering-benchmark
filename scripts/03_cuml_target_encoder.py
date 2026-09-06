"""기사 2번 경로: cuml.preprocessing.TargetEncoder 가 폴드/스무딩/조합을 대신 처리한다.

기사 예제 코드는 그대로 믿지 않는다. multi_feature_mode 같은 인자는
실제 설치된 버전에 있는지 시그니처로 먼저 확인하고, 없으면 있는 그대로 보고한다.
이 스크립트는 GPU 전용이다 (cuml 필요).
"""

import inspect

import cudf
import numpy as np
from cuml.preprocessing import TargetEncoder

from common import banner, demo_frame


def to_host(x):
    """cuML 출력은 GPU 객체다. numpy 로 내리려면 명시적으로 .to_numpy() 를 불러야 한다.

    np.asarray() 는 cudf 가 일부러 막아 둔다 (조용한 device->host 복사 방지).
    """
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    elif hasattr(x, "get"):  # cupy ndarray
        x = x.get()
    return np.asarray(x)


banner("설치된 cuml TargetEncoder 의 실제 시그니처")
sig = inspect.signature(TargetEncoder.__init__)
for name, param in sig.parameters.items():
    if name == "self":
        continue
    print(f"  {name} = {param.default!r}")
supports_combination = "multi_feature_mode" in sig.parameters
print(f"\nmulti_feature_mode 지원: {supports_combination}  (기사 주장: True)")

# 기사 예제와 같은 데이터를 cudf 로 올린다.
pdf = demo_frame()
gdf = cudf.from_pandas(pdf)  # 주의: cudf 26.08 에는 DataFrame.from_pandas 가 없다

# 학습/검증 분할 — 기사가 말한 대로 인코더를 학습 데이터에만 fit 한다.
train = gdf.iloc[:8].reset_index(drop=True)
valid = gdf.iloc[8:].reset_index(drop=True)
print(f"\ntrain {len(train)}행 / valid {len(valid)}행")

banner("① 단일 컬럼 (brand) — out-of-fold + 스무딩")
enc_kwargs = dict(n_folds=5, smooth=20, split_method="random", seed=42)
enc = TargetEncoder(**enc_kwargs)
tr_enc = enc.fit_transform(train[["brand"]], train["label"])
va_enc = enc.transform(valid[["brand"]])

out = train.to_pandas()[["brand", "label"]].copy()
out["brand_TE"] = to_host(tr_enc).ravel()
print(out.to_string(index=False))
print("\nvalid 인코딩:", to_host(va_enc).ravel())
print("같은 brand 인데 train 행마다 값이 다르다 = out-of-fold 가 실제로 동작한 것.")

banner("② 조합 인코딩 (brand × category)")
if supports_combination:
    enc2 = TargetEncoder(**enc_kwargs, multi_feature_mode="combination")
    cols = ["brand", "category"]
    tr2 = enc2.fit_transform(train[cols], train["label"])
    va2 = enc2.transform(valid[cols])
    arr = to_host(tr2)
    print("출력 shape:", arr.shape, "-> 컬럼 1개면 결합 인코딩, 2개면 독립 인코딩")
    o2 = train.to_pandas()[cols + ["label"]].copy()
    if arr.ndim == 1 or arr.shape[1] == 1:
        o2["brand_category_TE"] = arr.ravel()
    else:
        for i, c in enumerate(cols):
            o2[f"{c}_TE"] = arr[:, i]
    print(o2.to_string(index=False))
    print("\nvalid:", to_host(va2).ravel())
else:
    print("이 버전에는 multi_feature_mode 가 없다 — 기사 예제를 그대로 쓸 수 없다.")
    print("대안: brand_category 결합 컬럼을 직접 만들어 단일 컬럼으로 인코딩한다.")
    train2 = train.copy()
    valid2 = valid.copy()
    for d in (train2, valid2):
        d["brand_category"] = d["brand"].astype(str) + "_" + d["category"].astype(str)
    enc2 = TargetEncoder(**enc_kwargs)
    tr2 = enc2.fit_transform(train2[["brand_category"]], train2["label"])
    o2 = train2.to_pandas()[["brand", "category", "label"]].copy()
    o2["brand_category_TE"] = to_host(tr2).ravel()
    print(o2.to_string(index=False))

banner("③ 우리가 손으로 짠 것(02번)과의 차이")
print("직접 짜면: 폴드 분할 + 폴드별 groupby + 스무딩 + 미등장 범주 fallback 을 모두 관리해야 한다.")
print("cuML 은 이 전부를 estimator 하나로 처리하고, 계산은 GPU 에서 돈다.")
