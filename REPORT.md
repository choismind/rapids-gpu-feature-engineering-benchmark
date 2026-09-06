이 저장소의 `scripts/` 를 실제로 돌려 얻은 기록입니다. 실행 방법은 [README.md](README.md) 를 보십시오.

Medium 기사 「How Much of a Data Science Workflow Can Run on a GPU Today? Part 2: Feature Engineering」
(Parul Pandey, 2026-08-28) 의 주장을 이 PC 에서 직접 돌려 확인한 기록입니다.

기사는 개념 소개 글이라 **벤치마크 수치가 하나도 없습니다.** 그래서 CPU 대비 실측까지 함께 했습니다.
그 과정에서 **기사의 주장 하나가 사실과 다르다**는 것을 확인했습니다 — `cuml.accel` 이
scikit-learn 의 out-of-fold 동작을 조용히 버려서 타깃 누수를 되살립니다.

작업일: 2026-09-06

---

## 1. 배경 — 기사가 주장한 것

기사의 논지는 "피처 엔지니어링은 GPU 가속의 좋은 후보"입니다. 연산 하나가 무거워서가 아니라,
이 작업이 전처리 한 단계가 아니라 **탐색 문제**이기 때문입니다. 피처를 만들고 → 학습시키고 →
평가하고 → 바꿔서 다시 도는 반복이 본질이라, 한 사이클이 빨라지면 같은 시간에 더 많은
아이디어를 시험할 수 있다는 것입니다.

근거로 2025년 Kaggle Playground(배낭 가격 예측) 1위 Chris Deotte 의 사례를 듭니다.
**1만 개 이상의 피처를 생성·검증한 뒤 상위 500개를 선택**했고, 범주형 컬럼 8개를 쌍으로만
조합해도 후보 피처 28개가 추가로 생깁니다.

기사가 제시한 GPU 이전 경로는 두 가지입니다.

| 경로 | 도구 | 내용 |
|---|---|---|
| ① | `cuDF` / `cudf.pandas` | groupby 집계, 구간화, 컬럼 결합 등 데이터프레임 연산 |
| ② | `cuML` | 타깃 인코딩 등 sklearn 스타일 전처리기 |
| ③ | `cuml.accel` | sklearn 코드를 한 줄도 안 고치고 GPU 로 보내는 가속 레이어 |

검증 대상으로 삼은 구체적 주장은 다음 다섯 가지입니다.

1. 문법을 바꾸지 않고 `%load_ext cudf.pandas` 만으로 GPU 에서 돈다
2. 기사 예시 데이터에서 Acme 의 타깃 인코딩 값은 0.60, Solo 는 1.00 이다
3. 스무딩이 희소 범주를 전체 평균 쪽으로 끌어온다
4. `cuml.preprocessing.TargetEncoder` 의 `multi_feature_mode="combination"` 이 결합 인코딩을 준다
5. **`cuml.accel` 은 scikit-learn 의 동작을 유지한다**

---

## 2. 준비 — 환경 조사

### 2.1 하드웨어·드라이버

```powershell
nvidia-smi
```

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24GB (24463 MiB) |
| Compute capability | **12.0 (Blackwell, sm_120)** |
| 드라이버 / CUDA | 592.01 / CUDA 13.1 |

### 2.2 결정적 제약 — RAPIDS 는 Windows 에서 못 돕니다

PyPI 를 조회한 결과, RAPIDS 휠은 **manylinux 뿐이고 `win_amd64` 가 아예 없습니다.**

```
cudf_cu13-26.8.1-cp311-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
cuml_cu13-26.8.0-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

따라서 Windows 쪽 프로젝트 venv(Python 3.12.13) 에는 **설치 시도 자체가 무의미**합니다.
WSL2 를 거쳐야 합니다.

> [!Info] 검증 순서
> "설치해 보고 실패하면 그때 생각한다"가 아니라, **PyPI 의 휠 플랫폼 태그를 먼저 조회**해서
> 불가능을 확정한 뒤 대안을 물었습니다. 수 GB 다운로드를 낭비하지 않는 방법입니다.

### 2.3 WSL2 GPU 패스스루 확인

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "nvidia-smi -L; ls -l /dev/dxg; python3 --version"
```

```
GPU 0: NVIDIA GeForce RTX 5090 Laptop GPU (UUID: GPU-bceee3c1-...)
crw-rw-rw- 1 root root 10, 125 /dev/dxg
Python 3.12.3
```

`/dev/dxg` 가 있고 `nvidia-smi -L` 이 GPU 를 인식하므로 패스스루는 이미 동작합니다.
디스크 677GB 여유, RAM 47GB 로 여력도 충분했습니다.

### 2.4 선택지 제시

세 가지 중 사용자가 ①을 선택했습니다.

| 안 | 내용 | 비용 |
|---|---|---|
| **① WSL2 설치** | 스크립트는 Windows, 실행만 WSL | 다운로드 3~5GB |
| ② 스크립트만 작성 | CPU 로만 동작 확인 | 0, 단 핵심 미검증 |
| ③ Docker RAPIDS 이미지 | 격리 최상 | 이미지 10GB+ |

---

## 3. 설치

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "python3.12 -m venv ~/rapids-venv && ~/rapids-venv/bin/pip install 'cudf-cu13==26.8.*' 'cuml-cu13==26.8.*' scikit-learn pandas pyarrow"
```

venv 는 **WSL 홈(`~/rapids-venv`)** 에 두었습니다. `/mnt/d` 위에 두면 DrvFs I/O 때문에 느려집니다.
스크립트는 Windows 쪽에 두고 WSL 이 `/mnt/...` 로 읽습니다 (런처가 경로를 자동 변환합니다).

설치 결과:

| 패키지 | 버전 |
|---|---|
| cudf-cu13 | **26.08.01** |
| cuml-cu13 | **26.08.00** |
| cupy-cuda13x | 14.2.0 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| Python | 3.12.3 |

> [!Note] 기사와의 버전 차이
> 기사는 `pip install "cuml-cu13==26.6.*"` 를 씁니다. 현재 최신은 26.8 이라 그대로 적용했습니다.
> Blackwell(sm_120) 에서 별도 조치 없이 동작합니다.

---

## 4. 구현 — 검증 스크립트 설계

`scripts/` 에 12개 파일을 만들었습니다.

| 파일 | 역할 |
|---|---|
| `common.py` | 기사와 같은 합성 리뷰 데이터셋 + 백엔드 판정 헬퍼 |
| `00_env.py` | 설치·GPU 인식·백엔드 확인 |
| `01_cudf_features.py` | 기사 ①경로 — groupby 집계 / `pd.cut` / 컬럼 결합 |
| `02_target_encoding_manual.py` | 타깃 인코딩의 두 함정과 처방을 손으로 재현 |
| `03_cuml_target_encoder.py` | 기사 ②경로 — 네이티브 `cuml.preprocessing.TargetEncoder` |
| `04_cuml_accel_sklearn.py` | 기사 ③주장 검증 — `cuml.accel` 이 정말 동작을 보존하는가 |
| `05_benchmark_features.py` | 1000만 행 피처 생성 CPU vs GPU |
| `06_benchmark_target_encoder.py` | sklearn vs cuml TargetEncoder 정면 비교 |
| `07_realistic_pipeline.py` | 파이프라인 한 판 — 조합 28개 + OOF 인코딩, 크기별 교차점 |
| `08_benchmark_join_io.py` | 조인·파일 I/O (GPU 쓰기 크래시 우회 포함) |
| `run.ps1` / `run_all.ps1` | PowerShell 에서 WSL 로 넘기는 런처 |

### 4.1 설계 원칙 — 기사 예제 코드를 믿지 않는다

`03_cuml_target_encoder.py` 는 `multi_feature_mode` 를 그냥 쓰지 않고,
**설치된 버전의 시그니처를 먼저 조회해서 있는지 확인**한 뒤 없으면 대안 경로로 갑니다.

```python
sig = inspect.signature(TargetEncoder.__init__)
supports_combination = "multi_feature_mode" in sig.parameters
```

이 원칙 덕분에 아래 4.3 의 문제들이 조용히 넘어가지 않고 드러났습니다.

### 4.2 데이터셋 — 기사 숫자를 그대로 재현하도록 구성

`common.py` 의 `demo_frame()` 은 10행짜리로, 기사 본문의 산술이 그대로 나오게 짰습니다.

- Acme 5행, 그중 3행 `label=1` → 타깃 인코딩 **0.60**
- Solo 1행, `label=1` → **1.00** (희소 범주 문제)

`big_frame()` 은 규모 실험용으로, 기사가 말한 "10만 → 1000만 행"과
"고유값 10만 개짜리 고카디널리티 컬럼(`product_id`)"을 그대로 만듭니다.

### 4.3 기사 예제 코드를 그대로 쓸 수 없었던 지점 5곳

> [!Warning] 실행하면서 실제로 걸린 것들
> 1. **`%load_ext cudf.pandas` / `%load_ext cuml.accel` 은 IPython 매직입니다.**
>    `.py` 스크립트에서는 `python -m cudf.pandas script.py` 로 실행해야 합니다.
> 2. **`cudf.DataFrame.from_pandas` 는 cudf 26.08 에 없습니다.** `cudf.from_pandas(df)` 를 씁니다.
>    (`AttributeError: type object 'DataFrame' has no attribute 'from_pandas'`)
> 3. **cuML 출력에 `np.asarray()` 를 쓰면 `TypeError` 입니다.** 명시적으로 `.to_numpy()` 를 부릅니다.
>    조용한 device→host 복사를 막으려는 의도적 설계입니다.
> 4. **백엔드 판정에 `type(pd.DataFrame()).__module__` 은 못 씁니다.** 프록시가 `"pandas"` 로
>    위장합니다. `cudf.pandas.LOADED` 와 `_fsproxy_fast` 속성으로 판정해야 합니다.
>    — 제가 처음 이 함정에 빠져서, GPU 로 돌고 있는데 "CPU"로 표시되는 버그를 냈습니다.
> 5. **sklearn `TargetEncoder` 는 이진 타깃에 StratifiedKFold 를 씁니다.** 클래스당 표본이 `cv` 보다
>    적으면 `ValueError` 로 죽습니다. 기사의 8행 예제에 `cv=5` 를 그대로 넣으면 실패합니다.

---

## 5. 테스트 결과 — 기사가 맞았던 것

### 5.1 문법 무변경 (주장 1) — 사실

`01_cudf_features.py` 를 두 백엔드로 돌린 출력이 **완전히 동일**합니다.

```powershell
.\run.ps1 01_cudf_features.py        # pandas (CPU)
.\run.ps1 01_cudf_features.py -Gpu   # cudf.pandas (GPU)
```

세 연산 모두 코드 한 글자 안 바뀝니다.

```python
df["brand_avg_price"] = df.groupby("brand")["price"].transform("mean")
df["price_bin"] = pd.cut(df["price"], bins=[0, 100, 150, float("inf")],
                         labels=["low", "medium", "high"])
df["brand_category"] = df["brand"].astype(str) + "_" + df["category"].astype(str)
```

Acme 평균 가격 125.00 (= (80+120+250+45+130)/5) 도 양쪽에서 같습니다.

### 5.2 기사 본문 숫자 재현 (주장 2) — 사실

```
brand  brand_TE  n
 Acme       0.6  5
 Nova       0.5  4
 Solo       1.0  1
```

### 5.3 스무딩 효과 (주장 3) — 사실

`smoothed = (n × category_mean + smooth × prior) / (n + smooth)`, prior = 0.600

| brand | smooth=0 | 1 | 5 | 20 |
|---|---:|---:|---:|---:|
| Acme | 0.600 | 0.600 | 0.600 | 0.600 |
| Nova | 0.500 | 0.520 | 0.556 | 0.583 |
| **Solo** | **1.000** | 0.800 | 0.667 | **0.619** |

리뷰 1건짜리 Solo 가 전체 평균 쪽으로 끌려 내려옵니다.

### 5.4 out-of-fold 가 누수를 막는다 — 사실

5-fold OOF 를 손으로 구현하니 같은 브랜드인데 fold 마다 값이 갈립니다.
자기 라벨을 쓰지 않았다는 증거입니다.

```
brand  label  fold  brand_TE_naive  brand_TE_oof
 Acme      1     0             0.6      0.500000
 Acme      0     3             0.6      0.666667
 Nova      1     4             0.5      0.333333
 Solo      1     0             1.0      0.571429   ← 다른 폴드에 Solo 가 없어 prior 로 fallback
```

계산 횟수는 순진한 방식 1회 대 OOF 5회입니다. 피처가 F개면 F×5회로 불어납니다.
**기사가 말한 "작업량 폭증"이 여기서 눈으로 확인됩니다.**

### 5.5 결합 인코딩 (주장 4) — 사실, 게다가 기본값

cuml 26.08 의 실제 시그니처:

```
n_folds = 4
smooth = 0
seed = 42
split_method = 'interleaved'
stat = 'mean'
multi_feature_mode = 'combination'   ← 기본값이 이미 combination
```

`brand` + `category` 를 넣으면 출력 shape 이 `(8,)` — 결합 인코딩 1컬럼입니다.
반면 `cuml.accel` 경로의 sklearn 은 `(8, 2)` — 컬럼별 독립 인코딩입니다.
**기사가 지적한 차이는 정확합니다.**

---

## 6. 기사와 어긋난 것 — cuml.accel 은 sklearn 동작을 보존하지 않습니다

기사 원문은 이렇게 씁니다.

> cuml.accel keeps the scikit-learn behavior, so several input columns are encoded independently.

앞부분(컬럼 독립 인코딩)은 맞지만, **"keeps the scikit-learn behavior" 자체가 틀렸습니다.**

### 6.1 검증 방법

`TargetEncoder.fit_transform` 이 out-of-fold 인지 아닌지는 출력값으로 판정할 수 있습니다.

- **OOF 라면**: 같은 범주값이라도 어느 폴드에 속했느냐에 따라 인코딩 값이 달라집니다.
  `cv=5` 면 범주당 서로 다른 값이 5개 나와야 합니다.
- **OOF 가 아니라면**: 범주당 값이 1개뿐이고, 그 값은 전체 데이터로 계산한 스무딩값과 정확히 같습니다.

20,000행 / `cv=5` / `smooth=20` 으로 두 경로를 돌려 대조했습니다.

```powershell
.\run.ps1 04_cuml_accel_sklearn.py          # 순정 sklearn
.\run.ps1 04_cuml_accel_sklearn.py -Accel   # cuml.accel
```

### 6.2 결과

| | 범주당 서로 다른 인코딩 값 | 전체데이터 스무딩값과의 최대 차이 |
|---|---:|---:|
| 순정 sklearn (CPU) | **5** | 0.008567 |
| `cuml.accel` (GPU) | **1** | **0.000000** |

순정 sklearn 쪽 실제 출력:

```
brand
Acme     5
Kilo     5
Nova     5
...
>>> 최대 5개 = 폴드별로 값이 갈린다. out-of-fold 동작 확인.
```

`cuml.accel` 쪽:

```
brand
Acme     1
Kilo     1
Nova     1
...
>>> 값이 범주당 1개뿐 = out-of-fold 가 아니다. 타깃 누수가 그대로 남는다.

       fit_transform_첫값  전체데이터_스무딩값   차이
Acme           0.404093    0.404093            0.0
Kilo           0.517733    0.517733            0.0
...
최대 차이 = 0.000000
```

### 6.3 해석

차이가 **정확히 0** 이라는 것은, `fit_transform` 이 자기 행의 라벨까지 포함한
전체 데이터로 인코딩했다는 뜻입니다. 즉 `cv=5` 인자가 **조용히 무시**되고,
기사가 "OOF 로 막아야 한다"고 설명한 **타깃 누수가 그대로 되살아납니다.**

경고도 예외도 없습니다. 코드는 그대로 돌고 값만 달라집니다.

> [!Danger] 실무 함의
> `cuml.accel` 로 타깃 인코딩을 가속하면 **검증 점수가 낙관적으로 부풀 수 있습니다.**
> 학습 때는 좋아 보이는데 실제 데이터에서 무너지는 전형적인 형태입니다.
>
> 타깃 인코딩만큼은 `cuml.accel` 에 맡기지 말고
> `cuml.preprocessing.TargetEncoder` 를 **직접** 쓰는 편이 안전합니다.
>
> (cuml 26.08.00 기준. 상위 버전에서 고쳐졌는지는 재확인이 필요합니다.)

---

## 7. 벤치마크 — 기사에 없는 수치

### 7.1 피처 생성 (1000만 행, `product_id` 고유값 10만)

3회 반복 중 최소값, 워밍업 1회 제외.

| 연산 | CPU (pandas) | GPU (cudf.pandas) | 배속 |
|---|---:|---:|---:|
| `brand + "_" + category` 문자열 결합 | 1551.0 ms | 13.3 ms | **116.6×** |
| `pd.cut` 구간화 | 131.0 ms | 7.1 ms | **18.5×** |
| `groupby("product_id")["label"].mean()` | 152.6 ms | 9.5 ms | **16.1×** |
| `product_id.value_counts()` | 133.0 ms | 12.5 ms | **10.6×** |
| OOF 타깃 인코딩 5 folds (직접 구현) | 1929.3 ms | 494.7 ms | **3.9×** |
| `groupby("brand")` transform (8그룹) | 86.7 ms | 45.4 ms | **1.9×** |

**GPU 이득은 데이터 크기보다 그룹 수·문자열 연산량에 붙습니다.**
저카디널리티 groupby(8그룹)는 1.9배뿐인데 문자열 결합은 116배입니다.
기사가 예로 든 "8개 범주형 컬럼을 쌍으로 조합 → 28개 피처"가 정확히 116배 구간입니다.

### 7.2 타깃 인코딩 (200만 행, 고유값 10만, 5 folds)

| | 시간 | 배속 |
|---|---:|---:|
| CPU `sklearn.preprocessing.TargetEncoder` | 0.61 s | 1× |
| GPU `cuml.preprocessing.TargetEncoder` | 0.14 s | **4.4×** |

두 구현은 폴드 분할·스무딩 공식이 달라 값이 1:1로 같지 않습니다(상관계수 0.799).
"같은 종류의 작업" 비교로 읽어야 합니다.

### 7.3 측정 함정 — 결론이 뒤집혔던 일

> [!Bug] 처음 측정에서 GPU 가 5배 느리게 나왔습니다
> 첫 측정: CPU 0.64s vs **GPU 3.23s** → "GPU 가 5배 느리다"
>
> 원인은 워밍업을 **1000행**으로 한 것이었습니다. 실제 크기의 커널이 컴파일되지 않아
> 본 측정 첫 회에 컴파일 비용이 통째로 들어갔습니다.
>
> **전체 크기로 한 번 워밍업**(0.48s)한 뒤 재측정하니 0.14s 로 떨어졌고,
> 결론이 "4.4배 빠름"으로 완전히 뒤집혔습니다.
>
> GPU 벤치마크에서 첫 회를 측정에 넣으면 결론 자체가 반대로 나옵니다.
> 워밍업은 **본 측정과 같은 크기**로 해야 합니다.

---

### 7.4 현실적인 파이프라인 한 판 — 어디서부터 이득인가

앞의 7.1 은 연산 단위 마이크로벤치라 "실제로 얼마나 이득인가"에는 답하지 못합니다.
그래서 기사가 인용한 캐글 사례의 형태를 그대로 한 판 돌렸습니다 (`07_realistic_pipeline.py`).

**범주형 8컬럼 → 모든 쌍 조합 28개 → 각 조합마다 5-fold out-of-fold 타깃 인코딩**
= 피처 28개 생성, groupby 140회.

| 행 수 | CPU 전체 | GPU 전체 | 배속 | 피처 1개당 (CPU→GPU) |
|---:|---:|---:|---:|---|
| 500,000 | 4.21 s | 5.48 s | **0.77×** (GPU 가 느림) | 0.150 s → 0.196 s |
| 2,000,000 | 17.64 s | 5.72 s | **3.1×** | 0.630 s → 0.204 s |
| 10,000,000 | 113.55 s | 10.50 s | **10.8×** | 4.056 s → 0.375 s |

단계별로 쪼개면 원인이 보입니다.

| 단계 | 500k | 2M | 10M |
|---|---|---|---|
| 컬럼 결합 CPU | 1.56 s | 6.32 s | 39.31 s |
| 컬럼 결합 GPU | 0.27 s | 0.26 s | **0.50 s** |
| OOF 인코딩 CPU | 2.65 s | 11.32 s | 74.11 s |
| OOF 인코딩 GPU | 5.21 s | 5.46 s | 10.00 s |

> [!Tip] 여기서 읽어야 할 세 가지
> 1. **교차점은 약 100만 행입니다.** 50만 행에서는 GPU 가 오히려 30% 느립니다.
>    28개 결합 + groupby 140회를 띄우는 **고정 오버헤드가 약 5초**라, 그보다 작은 일감은
>    커널 실행 비용이 계산 비용을 덮어씁니다.
> 2. **GPU 시간은 데이터 크기에 거의 반응하지 않습니다.** 컬럼 결합은 50만이든 1000만이든
>    0.26~0.50초입니다. CPU 는 선형으로 늘어(1.56 → 39.31초) 격차가 계속 벌어집니다.
>    즉 **데이터가 클수록 이득이 커지는 구조**입니다.
> 3. **1000만 행에서 GPU 병목은 타깃 인코딩(10.5초 중 10.0초)입니다.** 계산량이 아니라
>    groupby 140회의 커널 실행 횟수에 묶여 있습니다. 직접 짠 루프 대신 네이티브
>    `cuml.TargetEncoder` 를 쓰면 더 줄어듭니다(7.2 에서 sklearn 대비 4.4배).
>    따라서 위 GPU 수치는 **상한이 아니라 하한**입니다.

체감으로 옮기면, 1000만 행에서 피처 배치 한 판이 **1분 54초 대 10.5초**입니다.
자리를 뜨느냐 마느냐가 갈리는 차이이고, 기사가 말한 "같은 시간에 더 많은 아이디어를
시험한다"가 실제로 성립하는 구간입니다.

### 7.5 조인과 파일 I/O

7.4 까지는 메모리 안 연산만 쟀습니다. 실제 업무에서는 파일을 읽고 마스터 테이블을 붙이는
단계가 큰 비중을 차지하므로 이것도 쟀습니다 (`08_benchmark_join_io.py`).

9.2절 범죄 분석 시나리오 형태 그대로입니다. 사실 테이블 1000만 행,
행정동 마스터 3,500행(문자열 키), 피의자 마스터 100만 행(정수 키), 연도별 parquet 10개.

#### 조인 — 25배 내외

| 작업 | CPU | GPU | 배속 |
|---|---:|---:|---:|
| 행정동 마스터 3,500행 조인 (문자열 키) | 0.76 s | 0.03 s | **25×** |
| 피의자 마스터 100만 행 조인 (정수 키) | 0.48 s | 0.02 s | **24×** |
| 둘 다 연달아 (실제 파이프라인 형태) | 1.36 s | 0.05 s | **27×** |

문자열 키든 정수 키든, 마스터가 3,500행이든 100만 행이든 비슷하게 25배 안팎입니다.
조인은 GPU 가 안정적으로 강한 영역입니다.

#### 파일 읽기 — CSV 가 압도적

page cache 가 더운 상태에서 3회 최소값입니다. 디스크 대역폭이 아니라 **디코딩 속도** 비교입니다.

| 작업 | CPU | GPU | 배속 |
|---|---:|---:|---:|
| CSV 읽기 (493 MiB, 단일 파일) | 3.41 s | 0.07 s | **49×** |
| parquet 읽기 (163 MiB, 단일 파일) | 0.16 s | 0.02 s | **8×** |
| 연도별 parquet 10개 읽어 concat | 0.17 s | 0.13 s | **1.3×** |
| parquet 읽기 — `/mnt/d` (DrvFs) | 0.48 s | 0.22 s | 2.2× |

읽어야 할 두 가지가 있습니다.

- **CSV 파싱이 GPU 의 최대 강점입니다(49배).** 공공데이터가 CSV 로 배포되는 경우가 많은데,
  그 최초 적재 단계가 통째로 사라집니다.
- **작은 파일 여러 개는 GPU 이점이 거의 없습니다(1.3배).** 같은 총량인데 파일을 10개로
  쪼개면 GPU 는 0.02초 → 0.13초로 6.5배 느려집니다(CPU 는 0.16 → 0.17초로 변화 없음).
  파일당 커널 실행 오버헤드 때문입니다. **GPU 를 쓸 거면 파일을 크게 합치는 편이 유리합니다.**

#### 파일 쓰기 — 우회 경로가 필요하고, 이득도 작습니다

| 작업 | CPU | GPU (버퍼 경유) | 배속 |
|---|---:|---:|---:|
| CSV 쓰기 | 8.33 s | 2.35 s | 3.5× |
| parquet 쓰기 | 1.19 s | 1.77 s | **0.67× — GPU 가 느림** |

> [!Bug] cudf 26.08 + WSL2 — GPU 파일 쓰기가 크래시합니다
> `df.to_parquet("파일경로")` 가 **`CUDA_ERROR_ILLEGAL_ADDRESS` 로 프로세스째 죽습니다.**
> OOM 이 아니라 불법 메모리 접근입니다.
>
> ```
> RuntimeError: CUDA error at:
>   /__w/kvikio/kvikio/cpp/include/kvikio/detail/posix_io.hpp:257:
>   CUDA_ERROR_ILLEGAL_ADDRESS(an illegal memory access was encountered)
> ```
>
> **격리 결과** — 깨진 것은 KvikIO 의 **쓰기 경로 하나**입니다.
>
> | 경로 | 결과 |
> |---|---|
> | `to_numpy` / `to_pandas` / `to_arrow` | ✅ 정상 |
> | `read_parquet` / `read_csv` | ✅ 정상 |
> | `to_parquet` / `to_csv` / `to_orc` (파일경로) | ❌ **전부 크래시** |
>
> 행 수·컬럼 타입 무관합니다. 100만 행 정수 2컬럼에서도 죽습니다.
> 환경변수 우회 8종(`KVIKIO_COMPAT_MODE` ON/on/TRUE, `KVIKIO_NTHREADS=1`,
> `LIBCUDF_CUFILE_POLICY` OFF/ALWAYS 등) **전부 실패**했습니다.
>
> **우회로**: 파일경로 대신 **버퍼에 쓰고 버퍼를 파일로 내립니다.** KvikIO 를 타지 않습니다.
>
> ```python
> import io
> buf = io.BytesIO()
> df.to_parquet(buf, index=False)          # 파일경로 대신 버퍼
> pathlib.Path(dst).write_bytes(buf.getvalue())
>
> pathlib.Path(dst).write_text(df.to_csv(index=False), encoding="utf-8")
> ```
>
> WSL2 에 GPUDirect Storage 가 없어서 KvikIO 의 POSIX 폴백 경로를 타는데,
> 그 경로가 깨진 것으로 보입니다. 네이티브 리눅스에서도 같은지는 확인하지 못했습니다.

우회로를 써도 parquet 쓰기는 CPU 보다 느립니다(1.77s 대 1.19s).
**쓰기는 GPU 로 옮길 값이 없습니다.** CSV 쓰기만 3.5배로 의미가 있습니다.

#### `/mnt/d` 를 쓰지 마십시오

같은 parquet 를 WSL 로컬 ext4 대신 `/mnt/d`(DrvFs)에서 읽으면,
CPU 는 3.1배, **GPU 는 8.9배** 느려집니다. GPU 는 디코딩이 워낙 빨라
파일시스템 오버헤드가 전체를 지배해 버립니다.

**GPU 로 처리할 데이터는 WSL 로컬 파일시스템에 두어야 합니다.**
Windows 볼륨에 둔 채로 돌리면 가속분을 DrvFs 가 그대로 먹습니다.

---

## 8. 재현 방법

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

.\run.ps1 07_realistic_pipeline.py       -Rows 10000000   # 파이프라인 한 판 CPU
.\run.ps1 07_realistic_pipeline.py -Gpu  -Rows 10000000   # 파이프라인 한 판 GPU

.\run.ps1 08_benchmark_join_io.py        -Rows 10000000   # 조인·파일 I/O CPU
.\run.ps1 08_benchmark_join_io.py  -Gpu  -Rows 10000000   # 조인·파일 I/O GPU
```

`08` 은 WSL 홈에 `~/bench_data`(약 1.5GB)를 만듭니다. 다 쓰면 지우십시오.

`run.ps1` 이 WSL2 의 `~/rapids-venv/bin/python` 으로 넘깁니다.
벤치마크 원본 JSON 은 `results/` 에 있습니다.

최종 확인: 8단계 전부 실행, **Traceback 0건**.

---

## 9. 정리

| 기사 주장 | 판정 |
|---|---|
| 문법 변경 없이 GPU 로 이전된다 | ✅ 사실 |
| Acme 0.60 / Solo 1.00 | ✅ 재현됨 |
| 스무딩이 희소 범주를 완화한다 | ✅ 사실 |
| OOF 가 타깃 누수를 막는다 | ✅ 사실 |
| `multi_feature_mode="combination"` 이 결합 인코딩을 준다 | ✅ 사실 (26.08 에선 기본값) |
| **`cuml.accel` 이 sklearn 동작을 보존한다** | ❌ **거짓 — `cv` 무시, OOF 소멸** |
| 피처 엔지니어링은 GPU 가속의 좋은 후보다 | ✅ 조건부 — 그룹 수·문자열 연산이 많을 때 |

기사 자체는 "왜 이 작업이 GPU 에 어울리는가"를 워크로드 구조로 잘 설명합니다.
다만 **예제 코드는 버전이 밀려 있고, `cuml.accel` 에 대한 설명은 위험한 오류**입니다.
가속 레이어를 쓸 때는 값이 같은지를 반드시 자기 데이터로 대조해야 합니다.

### 9.1 그래서 어디에 쓰면 이득인가

| 상황 | 판단 | 근거 |
|---|---|---|
| **100만 행 이상 + 범주형 조합 폭발** | **가장 큰 이득** | 컬럼 결합 GPU 시간이 크기와 무관(0.26~0.50s). 10M 에서 79× |
| **고카디널리티 그룹 집계** (product_id·user_id 수준) | **큰 이득** | 10M groupby 16×, value_counts 11× |
| **피처를 수십~수백 개 만들어 돌리는 반복 실험** | **큰 이득** | 10M 파이프라인 한 판 113.6s → 10.5s (10.8×) |
| **대용량 CSV 적재** | **가장 큰 이득** | 493 MiB CSV 읽기 3.41s → 0.07s (49×) |
| **마스터 테이블 조인** | **큰 이득** | 문자열 키 25×, 정수 키 24× (7.5절) |
| 100만 행 미만 | **쓰지 말 것** | 50만 행에서 GPU 가 0.77× — 고정 오버헤드 ~5s 를 못 넘김 |
| 저카디널리티 groupby 단발 (그룹 8개) | 무의미 | 1.9×, 절대 절감 41ms |
| 작은 파일 여러 개 읽기 | 거의 무의미 | 10개로 쪼개면 1.3× — 합쳐서 읽을 것 |
| **파일 쓰기** | **옮길 값 없음** | parquet 쓰기는 0.67× (GPU 가 느림). 게다가 크래시 우회 필요 |
| 데이터를 `/mnt/d` 에 둔 채 처리 | **하지 말 것** | DrvFs 가 GPU 를 8.9× 느리게 만듦 |
| 타깃 인코딩을 `cuml.accel` 로 | **금지** | 6절 — OOF 소멸, 타깃 누수 |

한 줄로 줄이면 **"큰 데이터에 문자열·고카디널리티 범주 연산을 많이, 반복해서 돌릴 때"** 입니다.
데이터가 작거나 연산 종류가 단순하면 GPU 로 옮기는 값을 못 합니다.

### 9.2 적용 사례 — 도심 범죄 분석

위 표의 "상황" 문구만으로는 실제 업무에 감이 안 잡히므로, 구체적인 분석 과제에 대응시켜 봅니다.

> **과제 가정**: 최근 10년 도심 지역별·범죄 유형별 범죄율과 범죄자 신상의 상호 연관성 조사

데이터 형태를 이렇게 가정합니다. 사건 1건 = 1행, 10년 전국이면 **1000만 행대**입니다.

| 종류 | 컬럼 예 | 고유값 |
|---|---|---|
| 고카디널리티 범주 | 피의자 가명 ID, 행정동 코드, 격자 좌표 | 수천 ~ 수백만 |
| 중간 범주 | 시군구, 죄명 세분류, 직업 | 수십 ~ 수백 |
| 저카디널리티 범주 | 범죄 대분류, 요일, 시간대 구간, 성별 | 2 ~ 20 |
| 수치 | 연령, 피해액, 전과 횟수 | — |

아래 배속은 전부 7절의 **실측값**이고, 업무 대응은 그 위에 얹은 해석입니다.

#### ① 교차 상호작용 탐색 — 79배 구간

분석가가 실제로 하는 일은 **크로스탭을 수십~수백 개 찍어 보는 것**입니다.
"강남구에서 절도가 심야에 20대에게 유독 높은가"는 결국
`지역 × 죄종 × 시간대 × 연령대` 조합의 집계입니다.

```python
df["동_죄종"]     = df["행정동"].astype(str) + "_" + df["죄종"].astype(str)
df["동_시간대"]   = df["행정동"].astype(str) + "_" + df["시간대"].astype(str)
df["죄종_연령대"] = df["죄종"].astype(str)   + "_" + df["연령대"].astype(str)
```

개별 컬럼만 봐서는 안 보이는 신호가 있기 때문에 만듭니다. 역삼동 전체 범죄율이 평균이고
절도 발생률도 평균인데 **`역삼동 × 심야 × 절도`만 튀는** 경우는 조합 컬럼을 만들어야 드러납니다.
기사가 인용한 캐글 우승자의 "신호가 약해서 피처 엔지니어링으로 모델이 찾도록 도와야 한다"가
이 얘기입니다.

조합 개수는 컬럼 수에 따라 급격히 불어납니다 (순서 무의미 — `동_죄종` = `죄종_동`).

| 범주형 컬럼 수 | 쌍 조합 C(n,2) | 3중 조합 C(n,3) | 합계 |
|---:|---:|---:|---:|
| 4 | 6 | 4 | 10 |
| 6 | 15 | 20 | 35 |
| **8** | **28** | **56** | **84** |
| 10 | 45 | 120 | 165 |
| 12 | 66 | 220 | 286 |

**조합 하나 = 집계 한 번**입니다. 1000만 행에서 조합 28개 생성이 **CPU 39.3초 대 GPU 0.50초**,
여기에 5-fold 타깃 인코딩까지 붙이면 총 113.6초 대 10.5초였습니다.
컬럼을 12개로 늘려 286개를 만들면 CPU 는 20분 가까이 걸립니다.

GPU 시간은 데이터 크기에 거의 반응하지 않으므로(0.26~0.50초),
**"조합을 몇 개까지 볼 것인가"를 시간이 아니라 분석 판단으로 정할 수 있게 됩니다.**

#### ② 행정동·피의자 단위 집계 — 16배 구간

- 행정동 3,500개 단위 10년치 인구 대비 범죄율
- 피의자 ID 수백만 개 단위 재범 횟수, 범행 간격, 죄종 전이

그룹이 수천~수백만 개인 groupby 입니다. 1000만 행에서 **groupby 16배, `value_counts` 11배**.
가명처리된 ID 도 고유값 많은 범주형 컬럼일 뿐이라 그대로 동작합니다.

#### ③ 지역을 모델 입력으로 바꾸기 — 타깃 인코딩

"재범 여부"를 타깃으로 두고 행정동을 넣으려면 원-핫은 3,500컬럼이 되므로 타깃 인코딩이 정석입니다.

> [!Danger] 6절의 금지 조항이 실제로 물리는 자리
> `cuml.accel` 로 가속하면 누수가 생겨, "이 동네는 재범률이 높다"는 값이
> **그 사건 자신의 재범 여부를 보고 만들어진 숫자**가 됩니다.
> 검증 점수는 좋게 나오는데 새 데이터에서 무너집니다.
> 범죄 분석은 결과가 치안 자원 배분 같은 정책 판단으로 이어지므로 대가가 특히 큽니다.
>
> 네이티브 `cuml.preprocessing.TargetEncoder` 를 쓰고 **스무딩을 반드시 걸 것.**
> 사건이 3건뿐인 행정동이 재범률 100% 로 잡히는 게 5.3절의 Solo 사례입니다.

#### ④ 가설을 스무 개 돌려 보는 루프 — 10.8배 구간

"소득분위와 절도", "유흥업소 밀도와 폭력범죄", "야간 유동인구와 성범죄" —
가설마다 피처 조합을 바꿔 집계를 다시 돌립니다.
1000만 행에서 **한 판이 1분 54초 대 10.5초**입니다.
오전 한나절에 시도할 수 있는 가설 수가 한 자릿수에서 두 자릿수로 바뀝니다.

#### ⑤ 반대로, 이 과제에서 GPU 를 쓰면 안 되는 작업

| 작업 | 이유 |
|---|---|
| 연도별 전국 범죄 건수 추이 | 그룹 10개짜리 groupby 한 번. 1.9배, 절감 41밀리초 |
| 광역시 1곳·1개년만 떼어 분석 | 수십만 행 구간. 50만 행에서 GPU 가 **0.77배로 오히려 느림** |
| 최종 집계표 시각화·통계검정 | 수천 행짜리 결과 테이블. CPU pandas 가 맞음 |

#### ⑥ 실무 배치 — 깔때기의 넓은 쪽만 GPU

```
1000만 행 원본 ──[GPU]──▶ 조합·집계·인코딩 ──▶ 수천 행 결과 ──[CPU]──▶ 검정·시각화·보고서
     (여기가 이득)                                    (여기부터는 GPU 가 손해)
```

#### ⑦ 판정 체크리스트

**2개 이상 해당하면 GPU 가 값을 합니다.**

1. 행이 100만 개 이상인가
2. 범주형 컬럼이 5개 이상이고 그 **조합**을 볼 것인가
3. 고유값 1,000개 이상인 컬럼(행정동·피의자 ID 등)으로 집계하는가
4. 같은 집계를 20번 이상 반복할 것인가

위 범죄 분석 과제는 네 가지 전부 해당해서, 실측 구간 중 가장 유리한 쪽에 있습니다.

#### ⑧ 데이터 적재 단계 (7.5절 측정 반영)

10년치를 파일에서 읽어 마스터를 붙이는 단계가 실제로는 첫 관문입니다. 여기서도 갈립니다.

| 적재 작업 | 판단 | 근거 |
|---|---|---|
| 범죄통계 원본 CSV 적재 | **가장 큰 이득 49×** | 공공데이터는 CSV 배포가 많음 |
| 행정동·피의자 마스터 조인 | **25× 내외** | 문자열 키·정수 키 모두 |
| 연도별 파일 10개로 나눠 읽기 | 1.3× — **한 파일로 합쳐 둘 것** | 파일당 커널 오버헤드 |
| 중간 산출물 parquet 저장 | 0.67× — **CPU 로 쓸 것** | 게다가 크래시 우회 필요 |

권장 형태는 이렇습니다.

1. 원본 CSV 를 **한 번만** GPU 로 읽어 → 단일 parquet 로 저장 (쓰기는 CPU 경로)
2. 그 parquet 를 **WSL 로컬 ext4** 에 둠 (`/mnt/d` 에 두면 GPU 가 8.9배 느려짐)
3. 이후 분석 반복은 그 파일을 GPU 로 읽어서 수행

---

## 참고문헌

- Parul Pandey, "How Much of a Data Science Workflow Can Run on a GPU Today? Part 2: Feature Engineering", *Medium*, 2026-08-28.
  <https://pandeyparul.medium.com/how-much-of-a-data-science-workflow-can-run-on-a-gpu-today-818890a2f0dc>

---

## 작성 도구

이 문서와 `scripts/` 의 검증 코드는 **Claude Code**(Anthropic)의 도움을 받아 작성했습니다.
환경 조사, 스크립트 구현, 벤치마크 실행, 결과 정리를 Claude Code 와 함께 진행했습니다.

본문의 모든 수치는 이 PC 에서 실제로 측정한 값이며, `scripts/` 를 그대로 돌려 재현할 수 있습니다.
