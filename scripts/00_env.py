"""설치 상태와 GPU 인식 여부를 있는 그대로 찍는다. 제일 먼저 돌릴 것."""

from common import banner

banner("환경")

import sys
print("python     :", sys.version.split()[0])

for mod in ("cudf", "cuml", "pandas", "sklearn", "numpy", "cupy"):
    try:
        m = __import__(mod)
        print(f"{mod:<11}: {getattr(m, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{mod:<11}: 실패 -> {type(exc).__name__}: {exc}")

banner("GPU")
try:
    import cupy
    dev = cupy.cuda.runtime.getDeviceProperties(0)
    free, total = cupy.cuda.runtime.memGetInfo()
    print("device     :", dev["name"].decode())
    print("compute cap:", f"{dev['major']}.{dev['minor']}")
    print("memory     :", f"{free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
except Exception as exc:  # noqa: BLE001
    print("cupy 로 GPU 조회 실패 ->", type(exc).__name__, exc)

banner("실제 GPU 연산 한 번")
try:
    import cudf
    s = cudf.Series([1, 2, 3, 4, 5])
    print("cudf.Series.sum() =", s.sum(), "(GPU 에서 계산)")
except Exception as exc:  # noqa: BLE001
    print("cudf 연산 실패 ->", type(exc).__name__, exc)

banner("현재 pandas 백엔드")
from common import active_backend
print(active_backend())
print()
print("힌트: `python 00_env.py` 는 CPU, `python -m cudf.pandas 00_env.py` 는 GPU 로 나와야 한다.")
