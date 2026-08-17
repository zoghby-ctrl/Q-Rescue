from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_initial_view_has_no_runtime_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not app.exception
    assert any(button.label == "LOAD DEMO SCENARIO" for button in app.button)
    assert any(button.label == "RUN Q-RESCUE" for button in app.button)

