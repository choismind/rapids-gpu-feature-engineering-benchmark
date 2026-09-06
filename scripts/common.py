"""기사에서 쓰인 합성 제품 리뷰 데이터셋 (UCSD Amazon review 를 느슨하게 흉내낸 것).

작은 데모 프레임은 기사 본문의 숫자를 그대로 재현하도록 구성했다.
  - Acme 는 5행, 그중 3행이 label=1  -> 타깃 인코딩 0.60
  - Solo 는 1행뿐이고 label=1        -> 타깃 인코딩 1.00 (희소 범주 문제)
"""

from __future__ import annotations

import os


def is_gpu_pandas() -> bool:
    """cudf.pandas 프록시가 활성인지 판정한다.

    주의: 프록시는 __module__ 을 "pandas" 로 위장하므로 모듈 이름으로는 구분되지 않는다.
    (실측 2026-09-06: `python -m cudf.pandas` 로 돌려도 __module__ 은 "pandas".)
    믿을 수 있는 신호는 cudf.pandas.LOADED 와 프록시 전용 속성 _fsproxy_fast 다.
    """
    import sys

    import pandas as pd

    loaded = getattr(sys.modules.get("cudf.pandas"), "LOADED", False)
    proxied = hasattr(type(pd.DataFrame()), "_fsproxy_fast")
    return bool(loaded and proxied)


def active_backend() -> str:
    """지금 pandas 가 CPU 인지 cudf.pandas(GPU) 인지 있는 그대로 보고한다."""
    import sys

    import pandas as pd

    loaded = getattr(sys.modules.get("cudf.pandas"), "LOADED", False)
    proxied = hasattr(type(pd.DataFrame()), "_fsproxy_fast")
    tag = "cudf.pandas (GPU)" if (loaded and proxied) else "pandas (CPU)"
    return f"{tag}  [cudf.pandas.LOADED={loaded}, proxy={proxied}]"


def demo_frame():
    """기사 예시와 같은 10행짜리 프레임."""
    import pandas as pd

    return pd.DataFrame(
        {
            "user_id":    [f"u{i:02d}" for i in range(1, 11)],
            "product_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
            "brand":      ["Acme", "Acme", "Acme", "Acme", "Acme",
                           "Nova", "Nova", "Nova", "Nova", "Solo"],
            "category":   ["Electronics", "Home", "Electronics", "Toys", "Home",
                           "Electronics", "Toys", "Home", "Electronics", "Toys"],
            "price":      [ 80.0, 120.0, 250.0,  45.0, 130.0,
                            99.0, 160.0,  75.0, 210.0, 140.0],
            "label":      [1, 0, 1, 1, 0,
                           1, 0, 0, 1, 1],
        }
    )


def big_frame(n_rows: int | None = None, n_products: int = 100_000, seed: int = 0):
    """규모 실험용 프레임.

    기사가 말한 '10만 -> 1000만 행' 과 '고유값 10만개짜리 고카디널리티 컬럼' 을 그대로 만든다.
    numpy 로 CPU 에서 만든 뒤 pandas 에 넣는다 (생성 비용은 측정 대상이 아니다).
    """
    import numpy as np
    import pandas as pd

    n_rows = n_rows or int(os.environ.get("N_ROWS", 10_000_000))
    rng = np.random.default_rng(seed)

    brands = np.array(["Acme", "Nova", "Solo", "Vega", "Kilo", "Orbit", "Pico", "Zeta"])
    cats = np.array(["Electronics", "Home", "Toys", "Books", "Garden", "Sports"])

    brand_idx = rng.integers(0, len(brands), n_rows)
    cat_idx = rng.integers(0, len(cats), n_rows)
    prod_idx = rng.integers(0, n_products, n_rows)

    # 라벨에 브랜드/카테고리별 약한 신호를 심어 둔다 (기사의 'weak signal' 상황).
    base = 0.35 + 0.03 * brand_idx + 0.02 * cat_idx
    label = (rng.random(n_rows) < np.clip(base, 0, 1)).astype("int8")

    return pd.DataFrame(
        {
            "product_id": prod_idx,                       # 고카디널리티 (기본 10만 고유값)
            "brand": pd.Categorical(brands[brand_idx]),
            "category": pd.Categorical(cats[cat_idx]),
            "price": rng.gamma(shape=2.0, scale=60.0, size=n_rows).astype("float32"),
            "label": label,
        }
    )


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
