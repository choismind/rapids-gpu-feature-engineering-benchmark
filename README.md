# RAPIDS GPU 피처 엔지니어링 — 실측 검증

Medium 기사 [How Much of a Data Science Workflow Can Run on a GPU Today? Part 2: Feature Engineering](https://pandeyparul.medium.com/how-much-of-a-data-science-workflow-can-run-on-a-gpu-today-818890a2f0dc)
(Parul Pandey, 2026-08-28) 의 주장을 실제로 돌려 검증한 기록과 재현 스크립트입니다.

기사에는 벤치마크 수치가 하나도 없습니다. 그래서 CPU 대비 실측까지 했고,
그 과정에서 **기사의 주장 하나가 사실과 다르다**는 것을 확인했습니다.

전체 보고서: **[REPORT.md](REPORT.md)**

---

## 핵심 발견

### 1. `cuml.accel` 은 scikit-learn 의 동작을 보존하지 않습니다

기사 원문은 "cuml.accel keeps the scikit-learn behavior" 라고 씁니다. 틀렸습니다.
`TargetEncoder.fit_transform` 에서 **`cv` 인자가 조용히 무시되고 out-of-fold 가 사라집니다.**
기사가 "OOF 로 막아야 한다"고 설명한 **타깃 누수가 그대로 되살아납니다.** 경고도 예외도 없습니다.

20,000행 / `cv=5` / `smooth=20`:

| | 범주당 서로 다른 인코딩 값 | 전체데이터 스무딩값과의 최대 차이 |
|---|---:|---:|
| 순정 scikit-learn (CPU) | **5** | 0.008567 |
| `cuml.accel` (GPU) | **1** | **0.000000** |

차이가 정확히 0 이라는 것은 자기 행의 라벨까지 포함해 인코딩했다는 뜻입니다.
재현: `.\run.ps1 04_cuml_accel_sklearn.py` 와 `.\run.ps1 04_cuml_accel_sklearn.py -Accel` 을 대조하십시오.

> 타깃 인코딩을 GPU 로 돌릴 때는 `cuml.accel` 에 맡기지 말고
> `cuml.preprocessing.TargetEncoder` 를 직접 쓰십시오. (cuml 26.08.00 기준)

### 2. cudf + WSL2 에서 GPU 파일 쓰기가 크래시합니다

`df.to_parquet("파일경로")` 가 `CUDA_ERROR_ILLEGAL_ADDRESS` 로 프로세스째 죽습니다.
OOM 이 아니라 KvikIO 쓰기 경로의 불법 메모리 접근입니다.

| 경로 | 결과 |
|---|---|
| `to_numpy` / `to_pandas` / `to_arrow` | 정상 |
| `read_parquet` / `read_csv` | 정상 |
| `to_parquet` / `to_csv` / `to_orc` (파일경로) | **전부 크래시** |

행 수·컬럼 타입 무관합니다(100만 행 정수 2컬럼에서도 죽음).
환경변수 우회 8종(`KVIKIO_COMPAT_MODE`, `KVIKIO_NTHREADS`, `LIBCUDF_CUFILE_POLICY` 등) 전부 실패했습니다.

**우회로** — 파일경로 대신 버퍼에 쓰면 KvikIO 를 타지 않습니다.

```python
import io, pathlib
buf = io.BytesIO()
df.to_parquet(buf, index=False)
pathlib.Path(dst).write_bytes(buf.getvalue())
```

### 3. 어디서 이득이고 어디서 손해인가

파이프라인 한 판(범주형 8컬럼 → 쌍 조합 28개 → 각각 5-fold 타깃 인코딩):

| 행 수 | CPU | GPU | 배속 |
|---:|---:|---:|---:|
| 500,000 | 4.21 s | 5.48 s | **0.77× — GPU 가 느림** |
| 2,000,000 | 17.64 s | 5.72 s | 3.1× |
| 10,000,000 | 113.55 s | 10.50 s | **10.8×** |

**교차점은 약 100만 행입니다.** 그보다 작으면 커널 실행 고정 오버헤드(약 5초)를 못 넘깁니다.

연산별 배속 (1000만 행):

| 작업 | 배속 |
|---|---:|
| CSV 읽기 (493 MiB) | **49×** |
| 문자열 컬럼 결합 | **117×** |
| 조인 (문자열 키 / 정수 키) | 25× / 24× |
| `pd.cut` 구간화 | 18× |
| 고카디널리티 groupby | 16× |
| parquet 읽기 | 8× |
| OOF 타깃 인코딩 5 folds | 3.9× |
| 저카디널리티 groupby (8그룹) | 1.9× |
| 작은 파일 10개 읽기 | 1.3× |
| **parquet 쓰기** | **0.67× — GPU 가 느림** |

한 줄로 줄이면 **"큰 데이터에 문자열·고카디널리티 범주 연산을 많이, 반복해서 돌릴 때"** 입니다.

---

## 환경

**RAPIDS 는 Windows 네이티브 휠이 없습니다.** PyPI 에 manylinux 휠만 있으므로 WSL2 가 필요합니다.

```
cudf_cu13-26.8.1-cp311-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
cuml_cu13-26.8.0-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

측정 환경입니다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24GB, compute capability **12.0 (Blackwell)** |
| 드라이버 / CUDA | 592.01 / CUDA 13.1 |
| WSL2 | Ubuntu 24.04.3 LTS, `/dev/dxg` 패스스루 |
| Python | 3.12.3 |
| 라이브러리 | cudf **26.08.01**, cuml **26.08.00**, cupy 14.2.0, pandas 3.0.3, scikit-learn 1.9.0 |

기사는 `cuml-cu13==26.6.*` 를 쓰지만 현재 최신은 26.8 입니다.

### 설치

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "python3.12 -m venv ~/rapids-venv && ~/rapids-venv/bin/pip install 'cudf-cu13==26.8.*' 'cuml-cu13==26.8.*' scikit-learn pandas pyarrow"
```

venv 는 **WSL 홈**에 두십시오. `/mnt/*` 위에 두면 DrvFs I/O 때문에 느려집니다.

---

## 실행

`scripts/run.ps1` 이 스크립트 자신의 위치를 WSL 경로로 변환해 넘깁니다. 경로 하드코딩은 없습니다.

```powershell
cd scripts

.\run_all.ps1                                    # 벤치마크 제외 전부 (8단계)

.\run.ps1 00_env.py                              # CPU 백엔드
.\run.ps1 00_env.py -Gpu                         # cudf.pandas
.\run.ps1 04_cuml_accel_sklearn.py               # 순정 sklearn
.\run.ps1 04_cuml_accel_sklearn.py -Accel        # cuml.accel — 위와 ③④절 대조

.\run.ps1 05_benchmark_features.py       -Rows 10000000
.\run.ps1 05_benchmark_features.py -Gpu  -Rows 10000000
.\run.ps1 06_benchmark_target_encoder.py -Rows 2000000
.\run.ps1 07_realistic_pipeline.py       -Rows 10000000
.\run.ps1 07_realistic_pipeline.py -Gpu  -Rows 10000000
.\run.ps1 08_benchmark_join_io.py        -Rows 10000000
.\run.ps1 08_benchmark_join_io.py  -Gpu  -Rows 10000000
```

배포판·인터프리터가 다르면 환경변수로 바꿉니다.

```powershell
$env:RAPIDS_WSL_DISTRO = 'Ubuntu-22.04'
$env:RAPIDS_WSL_PYTHON = '$HOME/myenv/bin/python'
```

WSL 안에서 직접 돌려도 됩니다.

```bash
cd scripts
~/rapids-venv/bin/python -m cudf.pandas 01_cudf_features.py
```

> `08` 은 WSL 홈에 `~/bench_data`(약 840MB)와 `results/_io_tmp/`(약 163MB)를 만듭니다. 다 쓰면 지우십시오.

---

## 구성

| 경로 | 내용 |
|---|---|
| `REPORT.md` | 전체 보고서 — 준비·설치·구현·검증·벤치마크·적용 사례 |
| `scripts/common.py` | 기사와 같은 합성 리뷰 데이터셋 + 백엔드 판정 |
| `scripts/00_env.py` | 설치·GPU 인식·백엔드 확인 |
| `scripts/01_cudf_features.py` | groupby 집계 / `pd.cut` 구간화 / 컬럼 결합 |
| `scripts/02_target_encoding_manual.py` | 타깃 인코딩의 두 함정과 처방을 손으로 재현 |
| `scripts/03_cuml_target_encoder.py` | 네이티브 `cuml.preprocessing.TargetEncoder` |
| `scripts/04_cuml_accel_sklearn.py` | **`cuml.accel` 이 동작을 보존하는지 검증** |
| `scripts/05_benchmark_features.py` | 연산별 CPU vs GPU |
| `scripts/06_benchmark_target_encoder.py` | sklearn vs cuml TargetEncoder |
| `scripts/07_realistic_pipeline.py` | 파이프라인 한 판, 크기별 교차점 |
| `scripts/08_benchmark_join_io.py` | 조인·파일 I/O (GPU 쓰기 크래시 우회 포함) |
| `results/` | 벤치마크 원본 JSON |

---

## 기사 예제 코드를 그대로 쓸 수 없는 지점

1. `%load_ext cudf.pandas` / `%load_ext cuml.accel` 은 **IPython 매직**입니다.
   `.py` 에서는 `python -m cudf.pandas script.py` 로 실행합니다.
2. `cudf.DataFrame.from_pandas` 는 cudf 26.08 에 **없습니다**. `cudf.from_pandas(df)` 를 씁니다.
3. cuML 출력에 `np.asarray()` 를 쓰면 `TypeError` 입니다. `.to_numpy()` 를 명시적으로 부릅니다.
4. 백엔드 판정에 `type(pd.DataFrame()).__module__` 은 못 씁니다 — 프록시가 `"pandas"` 로 위장합니다.
   `cudf.pandas.LOADED` 와 `_fsproxy_fast` 로 판정합니다.
5. sklearn `TargetEncoder` 는 이진 타깃에 StratifiedKFold 를 씁니다. 클래스당 표본이 `cv` 보다
   적으면 `ValueError` 로 죽습니다. 기사의 8행 예제에 `cv=5` 를 그대로 넣으면 실패합니다.

## 측정 시 주의

- **GPU 워밍업은 본 측정과 같은 크기로 하십시오.** 1000행 워밍업으로 쟀을 때
  "GPU 가 5배 느리다"(3.23s vs 0.64s)는 반대 결론이 나왔고, 전체 크기로 워밍업하니
  0.14s(4.4배 빠름)로 뒤집혔습니다.
- 파일 읽기 수치는 page cache 가 더운 상태입니다. 디스크 대역폭이 아니라 **디코딩 속도** 비교입니다.
- **데이터를 `/mnt/*` 에 두지 마십시오.** 같은 parquet 를 DrvFs 에서 읽으면
  CPU 는 3.1배, GPU 는 **8.9배** 느려집니다.

---

## 참고문헌

- Parul Pandey, "How Much of a Data Science Workflow Can Run on a GPU Today? Part 2: Feature Engineering", *Medium*, 2026-08-28.
  <https://pandeyparul.medium.com/how-much-of-a-data-science-workflow-can-run-on-a-gpu-today-818890a2f0dc>
