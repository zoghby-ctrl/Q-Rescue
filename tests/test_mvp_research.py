from __future__ import annotations

from pathlib import Path

import pytest

from ui.research_view import load_research_findings


def test_research_headlines_are_loaded_from_validated_artifacts() -> None:
    findings = load_research_findings()
    assert findings["cross_id"].value == pytest.approx(0.912202, abs=1e-6)
    assert findings["traffic_shift"].value == pytest.approx(0.474702, abs=1e-6)
    assert findings["combined_shift"].value == pytest.approx(0.503720, abs=1e-6)
    assert findings["lono"].value == pytest.approx(0.900463, abs=1e-6)
    assert findings["real_r2"].value == pytest.approx(-0.186632, abs=1e-6)
    assert findings["real_coverage"].value == pytest.approx(0.544271, abs=1e-6)
    assert findings["xy_k8_p3"].value == pytest.approx(1.0, abs=1e-10)
    assert findings["x_k8_p3"].value == pytest.approx(0.022861, abs=1e-6)
    assert findings["dynamic_expected"].value == pytest.approx(0.028571, abs=1e-6)
    assert findings["dynamic_qaoa"].value == pytest.approx(0.171429, abs=1e-6)


def test_theme_source_contains_reduced_motion_and_focus_support() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "ui/theme.py").read_text(encoding="utf-8")
    components = (root / "ui/components.py").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in source
    assert "(pointer: coarse)" in source
    assert "focus-visible" in source
    assert "requestAnimationFrame" in source
    assert 'addEventListener("pointermove"' in source
    assert "pointer-events: none" in source
    assert 'getElementById("q-pointer-glow")' in source
    assert "Validated Research MVP" in components
    assert "Step 9.5" not in components
