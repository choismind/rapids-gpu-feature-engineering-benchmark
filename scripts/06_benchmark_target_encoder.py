"""sklearn TargetEncoder (CPU) vs cuml TargetEncoder (GPU) 를 한 프로세스에서 정면 비교한다.

cuml.accel 은 여기서 켜지 않는다 — 켜면 sklearn 쪽도 GPU 로 가서 비교가 안 된다.
    python 06_benchmark_target_encoder.py
"""

import os
import time

import numpy as np
import pandas as pd

from common import banner, big_frame

N_ROWS = int(os.environ.get("N_ROWS", 2_000_000))
N_PRODUCTS = int(os.environ.get("N_PRODUCTS", 100_000))
N_FOLDS = 5
SMOOTH = 20

banner("설정")
print(f"행 {N_ROWS:,} / product_id 고유값 {N_PRODUCTS:,} / folds {N_FOLDS} / smooth {SMOOTH}")

df = big_frame(N_ROWS, N_PRODUCTS)
y = df["label"].to_numpy()

banner("CPU — sklearn.preprocessing.TargetEncoder")
from sklearn.preprocessing import TargetEncoder as SkTE

REPEAT = int(os.environ.get("REPEAT", 3))

X_cpu = df[["product_id"]].astype("int64")
cpu_times = []
for i in range(REPEAT):
    t = time.perf_counter()
    sk = SkTE(smooth=SMOOTH, cv=N_FOLDS, shuffle=True, random_state=42)
    enc_cpu = sk.fit_transform(X_cpu, y)
    cpu_times.append(time.perf_counter() - t)
    print(f"  run {i+1}: {cpu_times[-1]:7.2f} s")
cpu_s = min(cpu_times)
print(f"  최소 {cpu_s:.2f} s   shape={np.asarray(enc_cpu).shape}")

banner("GPU — cuml.preprocessing.TargetEncoder")
import cudf
from cuml.preprocessing import TargetEncoder as CuTE

gdf = cudf.DataFrame({"product_id": df["product_id"].to_numpy(),
                      "label": df["label"].to_numpy()})

# 워밍업: 전체 크기로 한 번 돌려 커널 컴파일/할당 비용을 측정에서 뺀다.
# (1000행짜리 워밍업으로는 부족했다 — 실측에서 첫 회만 유독 느렸다.)
t = time.perf_counter()
CuTE(n_folds=N_FOLDS, smooth=SMOOTH, split_method="random", seed=42).fit_transform(
    gdf[["product_id"]], gdf["label"]
)
print(f"  워밍업(측정 제외): {time.perf_counter() - t:.2f} s")

gpu_times = []
for i in range(REPEAT):
    t = time.perf_counter()
    cu = CuTE(n_folds=N_FOLDS, smooth=SMOOTH, split_method="random", seed=42)
    enc_gpu = cu.fit_transform(gdf[["product_id"]], gdf["label"])
    _ = float(enc_gpu.iloc[0]) if hasattr(enc_gpu, "iloc") else float(enc_gpu[0])
    gpu_times.append(time.perf_counter() - t)
    print(f"  run {i+1}: {gpu_times[-1]:7.2f} s")
gpu_s = min(gpu_times)
arr = np.asarray(enc_gpu.to_numpy() if hasattr(enc_gpu, "to_numpy") else enc_gpu)
print(f"  최소 {gpu_s:.2f} s   shape={arr.shape}")

banner("결과")
print(f"  CPU (sklearn) : {cpu_s:8.2f} s")
print(f"  GPU (cuml)    : {gpu_s:8.2f} s")
print(f"  배속          : {cpu_s / gpu_s:8.2f}x")
print()
print("주의: 두 구현은 폴드 분할 방식·스무딩 공식이 완전히 같지 않다.")
print("      값이 1:1로 일치하지는 않으므로 '같은 종류의 작업' 비교로 읽을 것.")
a, b = np.asarray(enc_cpu).ravel(), arr.ravel()
n = min(len(a), len(b))
print(f"      상관계수(참고): {np.corrcoef(a[:n], b[:n])[0, 1]:.4f}")
