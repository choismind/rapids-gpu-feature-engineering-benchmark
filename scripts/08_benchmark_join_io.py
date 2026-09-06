"""조인과 파일 I/O 를 CPU vs GPU 로 잰다.

07번까지는 메모리 안 연산만 쟀다. 실제 업무에서는 파일을 읽고 마스터 테이블을 붙이는
단계가 큰 비중을 차지한다. 9.2절 범죄 분석 시나리오의 형태를 그대로 쓴다.

  - 연도별 parquet 10개를 읽어 붙이기
  - 행정동 마스터(3,500행, 문자열 키) 조인
  - 피의자 마스터(100만행, 정수 키) 조인
  - parquet / CSV 읽기·쓰기
  - WSL 로컬 ext4 vs /mnt/d (DrvFs) 비교

    python                08_benchmark_join_io.py   # CPU
    python -m cudf.pandas 08_benchmark_join_io.py   # GPU

주의: 파일 읽기는 3회 중 최소값이라 **page cache 가 더운 상태**의 수치다.
디스크 대역폭이 아니라 디코딩 속도 비교로 읽어야 한다.
"""

import io
import json
import os
import pathlib
import shutil
import time

import numpy as np
import pandas as pd

from common import active_backend, banner, is_gpu_pandas

N_ROWS = int(os.environ.get("N_ROWS", 10_000_000))
N_YEARS = 10
N_DONG = 3_500
N_OFFENDERS = 1_000_000

LOCAL_DIR = pathlib.Path(os.environ.get("BENCH_DIR", os.path.expanduser("~/bench_data")))

# DrvFs 비교용 — 이 스크립트가 Windows 볼륨(/mnt/*)에 있을 때만 의미가 있다.
# 저장소를 WSL 홈에 두고 돌리면 비교 대상이 없으므로 이 절은 건너뛴다.
MNT_DIR = pathlib.Path(
    os.environ.get("BENCH_MNT_DIR", pathlib.Path(__file__).resolve().parent / "results" / "_io_tmp")
)
MNT_AVAILABLE = str(MNT_DIR).startswith("/mnt/")

banner(f"백엔드: {active_backend()}")
print(f"행 {N_ROWS:,} / 연도 파일 {N_YEARS}개 / 행정동 {N_DONG:,} / 피의자 {N_OFFENDERS:,}")


def timed(name, fn, repeat=3, unit="s"):
    """워밍업 1회 후 최소값. 결과를 건드려 지연 평가를 막는다."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        if out is not None:
            _ = len(out) if hasattr(out, "__len__") else out
        best = min(best, time.perf_counter() - t)
    shown = best if unit == "s" else best * 1000
    print(f"  {name:<44} {shown:9.2f} {unit}")
    return best


# ------------------------------------------------------------------ 데이터 준비
rng = np.random.default_rng(0)
dong_vals = np.array([f"D{i:05d}" for i in range(N_DONG)])
crime_vals = np.array(["절도", "폭행", "사기", "성범죄", "마약", "방화",
                       "손괴", "횡령", "협박", "도박"])
hour_vals = np.array(["새벽", "오전", "낮", "저녁", "밤", "심야"])

fact = pd.DataFrame({
    "case_id": np.arange(N_ROWS, dtype="int64"),
    "year": rng.integers(2016, 2016 + N_YEARS, N_ROWS).astype("int16"),
    "dong": dong_vals[rng.integers(0, N_DONG, N_ROWS)],
    "crime": crime_vals[rng.integers(0, len(crime_vals), N_ROWS)],
    "hour_bucket": hour_vals[rng.integers(0, len(hour_vals), N_ROWS)],
    "offender_id": rng.integers(0, N_OFFENDERS, N_ROWS).astype("int64"),
    "amount": rng.gamma(2.0, 500.0, N_ROWS).astype("float32"),
    "label": (rng.random(N_ROWS) < 0.4).astype("int8"),
})

dong_master = pd.DataFrame({
    "dong": dong_vals,
    "population": rng.integers(1_000, 80_000, N_DONG).astype("int32"),
    "area_km2": rng.gamma(2.0, 1.5, N_DONG).astype("float32"),
})

offender_master = pd.DataFrame({
    "offender_id": np.arange(N_OFFENDERS, dtype="int64"),
    "prior_count": rng.integers(0, 12, N_OFFENDERS).astype("int8"),
    "income_decile": rng.integers(1, 11, N_OFFENDERS).astype("int8"),
})

results = {}

# ------------------------------------------------------------------ 조인
banner("조인 (merge)")

results["join_small_strkey"] = timed(
    f"① 행정동 마스터 {N_DONG:,}행, 문자열 키",
    lambda: fact.merge(dong_master, on="dong", how="left"),
)
results["join_mid_intkey"] = timed(
    f"② 피의자 마스터 {N_OFFENDERS:,}행, 정수 키",
    lambda: fact.merge(offender_master, on="offender_id", how="left"),
)
results["join_chained"] = timed(
    "③ 둘 다 연달아 (실제 파이프라인 형태)",
    lambda: fact.merge(dong_master, on="dong", how="left")
               .merge(offender_master, on="offender_id", how="left"),
)

# ------------------------------------------------------------------ 파일 I/O
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
single_pq = LOCAL_DIR / "fact.parquet"
single_csv = LOCAL_DIR / "fact.csv"
yearly_dir = LOCAL_DIR / "yearly"
yearly_dir.mkdir(exist_ok=True)

# --- GPU 쓰기 크래시 우회 -------------------------------------------------
# cudf 26.08 + WSL2 에서 df.to_parquet(파일경로) 는 KvikIO 쓰기 경로에서
# CUDA_ERROR_ILLEGAL_ADDRESS 로 죽는다 (to_csv, to_orc 도 동일. 읽기는 정상).
# 환경변수 우회(KVIKIO_COMPAT_MODE, LIBCUDF_CUFILE_POLICY 등) 8종 전부 실패했다.
# 버퍼로 쓰면 KvikIO 를 타지 않아 정상 동작하고, 오히려 CPU 보다 빠르다.
GPU = is_gpu_pandas()


def write_parquet(df, path):
    if not GPU:
        df.to_parquet(path, index=False)
        return
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    path.write_bytes(buf.getvalue())


def write_csv(df, path):
    if not GPU:
        df.to_csv(path, index=False)
        return
    path.write_text(df.to_csv(index=False), encoding="utf-8")


banner("파일 쓰기 (WSL 로컬 ext4)")
if GPU:
    print("  ※ GPU 는 파일경로 직접 쓰기가 크래시하므로 버퍼 경유 경로로 측정한다 (아래 주석 참조)")

results["write_parquet"] = timed(
    "④ parquet 쓰기 (단일 파일)",
    lambda: write_parquet(fact, single_pq),
    repeat=2,
)
print(f"     -> {single_pq.stat().st_size / 2**20:,.0f} MiB")

results["write_csv"] = timed(
    "⑤ CSV 쓰기 (단일 파일)",
    lambda: write_csv(fact, single_csv),
    repeat=2,
)
print(f"     -> {single_csv.stat().st_size / 2**20:,.0f} MiB")

# 연도별 파일 10개 (준비는 측정 제외)
if not (yearly_dir / f"y{2016 + N_YEARS - 1}.parquet").exists():
    for y in range(2016, 2016 + N_YEARS):
        part = fact[fact["year"] == y]
        write_parquet(part, yearly_dir / f"y{y}.parquet")
yearly_files = sorted(yearly_dir.glob("*.parquet"))
yearly_mib = sum(f.stat().st_size for f in yearly_files) / 2**20
print(f"\n연도별 파일 {len(yearly_files)}개 준비됨 ({yearly_mib:,.0f} MiB)")

banner("파일 읽기 (WSL 로컬 ext4, page cache 더운 상태)")
results["read_parquet"] = timed(
    "⑥ parquet 읽기 (단일 파일)", lambda: pd.read_parquet(single_pq)
)
results["read_csv"] = timed(
    "⑦ CSV 읽기 (단일 파일)", lambda: pd.read_csv(single_csv)
)
results["read_yearly_concat"] = timed(
    f"⑧ 연도별 parquet {N_YEARS}개 읽어 concat",
    lambda: pd.concat([pd.read_parquet(f) for f in yearly_files], ignore_index=True),
)

# ------------------------------------------------------------------ DrvFs 비교
banner("DrvFs 비교 — 같은 parquet 를 Windows 볼륨에서")
if not MNT_AVAILABLE:
    print(f"  건너뜀 — {MNT_DIR} 가 /mnt/* 아래가 아닙니다.")
    print("  Windows 볼륨과 비교하려면 BENCH_MNT_DIR 로 /mnt/... 경로를 지정하세요.")
else:
    MNT_DIR.mkdir(parents=True, exist_ok=True)
    mnt_pq = MNT_DIR / "fact.parquet"
    if not mnt_pq.exists():
        shutil.copy2(single_pq, mnt_pq)

    results["read_parquet_drvfs"] = timed(
        f"⑨ parquet 읽기 ({MNT_DIR.parts[2] if len(MNT_DIR.parts) > 2 else 'mnt'} 볼륨)",
        lambda: pd.read_parquet(mnt_pq),
    )
    ratio = results["read_parquet_drvfs"] / results["read_parquet"]
    print(f"     -> ext4 대비 {ratio:.1f}배")

# ------------------------------------------------------------------ 저장
banner("요약")
for k, v in results.items():
    print(f"  {k:<24} {v:8.2f} s")

backend = "gpu" if is_gpu_pandas() else "cpu"
out_dir = pathlib.Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
path = out_dir / f"joinio_{backend}_{N_ROWS}.json"
path.write_text(json.dumps({"backend": backend, "n_rows": N_ROWS,
                            "seconds": results}, indent=2), encoding="utf-8")
print(f"\n저장: {path}")
