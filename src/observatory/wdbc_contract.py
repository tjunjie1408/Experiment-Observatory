"""Canonical names shared by WDBC preparation and artifact validation."""

FEATURE_NAMES = tuple(
    f"{name}_{group}"
    for group in ("mean", "se", "worst")
    for name in (
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    )
)
