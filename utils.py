def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def fmt_multiple(v: float) -> str:
    """Format ROI as a multiple (e.g., 3.5x)."""
    return f"{v:.1f}x"
