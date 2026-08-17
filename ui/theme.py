"""Q-Rescue visual tokens and browser-only ambient presentation effects."""

from __future__ import annotations

import streamlit as st


GRAPHITE = "#0b0d10"
PANEL = "#111419"
PANEL_RAISED = "#15191f"
BORDER = "#29303a"
TEXT = "#f2f5f7"
MUTED = "#939da8"
CYAN = "#38d8ff"
CYAN_SOFT = "#153946"
BLUE = "#4f7cff"
VIOLET = "#8b5cf6"
EMBER = "#ff8a45"
EMBER_SOFT = "#42251a"


def _install_pointer_glow() -> None:
    """Install one idempotent client-side pointer glow in the parent page.

    The component never sends pointer coordinates to Python or Streamlit. It
    keeps one pending coordinate pair in the browser and applies it at most
    once per animation frame. A static CSS background remains if JavaScript
    is unavailable.
    """

    st.html(
        """
        <script>
        (() => {
          try {
            const host = window;
            const doc = document;
            const controllerKey = "__qRescuePointerGlowController";

            if (host[controllerKey] && typeof host[controllerKey].destroy === "function") {
              host[controllerKey].destroy();
            }

            let overlay = doc.getElementById("q-pointer-glow");
            if (!overlay) {
              overlay = doc.createElement("div");
              overlay.id = "q-pointer-glow";
              overlay.setAttribute("aria-hidden", "true");
              overlay.dataset.qrescueGlow = "browser-only";
              doc.body.appendChild(overlay);
            }

            const reducedMotion = host.matchMedia("(prefers-reduced-motion: reduce)");
            const coarsePointer = host.matchMedia("(pointer: coarse)");
            let frame = 0;
            let pointerX = host.innerWidth * 0.5;
            let pointerY = host.innerHeight * 0.35;

            const disabled = () => reducedMotion.matches || coarsePointer.matches;

            const updateAvailability = () => {
              const isDisabled = disabled();
              overlay.hidden = isDisabled;
              overlay.classList.toggle("is-active", !isDisabled);
              if (isDisabled && frame) {
                host.cancelAnimationFrame(frame);
                frame = 0;
              }
            };

            const paint = () => {
              frame = 0;
              doc.documentElement.style.setProperty("--q-pointer-x", `${pointerX}px`);
              doc.documentElement.style.setProperty("--q-pointer-y", `${pointerY}px`);
            };

            const onPointerMove = (event) => {
              if (disabled() || event.pointerType === "touch") return;
              pointerX = event.clientX;
              pointerY = event.clientY;
              if (!frame) frame = host.requestAnimationFrame(paint);
            };

            const onVisibilityChange = () => {
              if (doc.hidden && frame) {
                host.cancelAnimationFrame(frame);
                frame = 0;
              }
            };

            const addMediaListener = (media, listener) => {
              if (media.addEventListener) media.addEventListener("change", listener);
              else media.addListener(listener);
            };
            const removeMediaListener = (media, listener) => {
              if (media.removeEventListener) media.removeEventListener("change", listener);
              else media.removeListener(listener);
            };

            host.addEventListener("pointermove", onPointerMove, { passive: true });
            doc.addEventListener("visibilitychange", onVisibilityChange, { passive: true });
            addMediaListener(reducedMotion, updateAvailability);
            addMediaListener(coarsePointer, updateAvailability);
            updateAvailability();
            if (!disabled()) paint();

            host[controllerKey] = {
              destroy() {
                host.removeEventListener("pointermove", onPointerMove);
                doc.removeEventListener("visibilitychange", onVisibilityChange);
                removeMediaListener(reducedMotion, updateAvailability);
                removeMediaListener(coarsePointer, updateAvailability);
                if (frame) host.cancelAnimationFrame(frame);
              },
              status() {
                return {
                  disabled: disabled(),
                  framePending: Boolean(frame),
                  overlayCount: doc.querySelectorAll("#q-pointer-glow").length,
                };
              },
            };
          } catch (_) {
            // The static layered gradients remain the deliberate safe fallback.
          }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def apply_theme() -> None:
    """Apply restrained research-console styling and the safe ambient layer."""

    st.markdown(
        f"""
        <style>
        :root {{
            --q-bg: {GRAPHITE};
            --q-panel: {PANEL};
            --q-panel-raised: {PANEL_RAISED};
            --q-border: {BORDER};
            --q-text: {TEXT};
            --q-muted: {MUTED};
            --q-cyan: {CYAN};
            --q-blue: {BLUE};
            --q-violet: {VIOLET};
            --q-ember: {EMBER};
            --q-pointer-x: 50vw;
            --q-pointer-y: 32vh;
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }}

        html, body, #root {{
            height: 100%;
            min-height: 100%;
            background-color: var(--q-bg);
        }}

        #root {{
            position: relative;
            z-index: 1;
        }}

        body {{
            background-image:
                radial-gradient(70rem 46rem at -8% -10%, rgba(56, 216, 255, 0.075), transparent 64%),
                radial-gradient(62rem 42rem at 108% 4%, rgba(139, 92, 246, 0.062), transparent 66%),
                radial-gradient(54rem 36rem at 52% 112%, rgba(79, 124, 255, 0.045), transparent 68%),
                linear-gradient(145deg, #090b0e 0%, #0b0d10 48%, #0a0c10 100%);
            background-attachment: fixed;
        }}

        #q-pointer-glow {{
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            opacity: 0;
            background:
                radial-gradient(
                    42rem circle at var(--q-pointer-x) var(--q-pointer-y),
                    rgba(56, 216, 255, 0.075) 0%,
                    rgba(79, 124, 255, 0.048) 34%,
                    rgba(139, 92, 246, 0.026) 52%,
                    transparent 72%
                );
            transition: opacity 160ms ease;
            will-change: background;
        }}

        #q-pointer-glow.is-active {{ opacity: 1; }}
        #q-pointer-glow[hidden] {{ display: none !important; }}

        .stApp {{
            background: transparent;
            color: var(--q-text);
        }}

        .block-container {{
            max-width: 1760px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4, p, label {{ color: var(--q-text); }}

        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] small {{ color: var(--q-muted); }}

        [data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        [data-testid="stMetric"] {{
            border-color: var(--q-border) !important;
            border-radius: 10px !important;
        }}

        [data-testid="stForm"] {{
            background: linear-gradient(180deg, rgba(21, 25, 31, 0.88), rgba(17, 20, 25, 0.96));
            padding: 0.95rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }}

        [data-testid="stMetric"] {{
            background: linear-gradient(155deg, rgba(21, 25, 31, 0.88), rgba(17, 20, 25, 0.96));
            border: 1px solid var(--q-border);
            padding: 0.72rem 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
            transition: border-color 140ms ease, background-color 140ms ease;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            border: 1px solid #3a4652;
            border-radius: 8px;
            min-height: 2.55rem;
            font-weight: 650;
            transition: border-color 140ms ease, box-shadow 140ms ease,
                        background-color 140ms ease, transform 140ms ease;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border-color: #557084;
            background-color: rgba(21, 29, 36, 0.98);
        }}

        .stButton > button[kind^="primary"],
        .stFormSubmitButton > button[kind^="primary"] {{
            color: #041014;
            background: linear-gradient(110deg, #6ae7ff 0%, var(--q-cyan) 56%, #75cfff 100%);
            border-color: #85ebff;
            box-shadow: 0 0 0 1px rgba(56, 216, 255, 0.20),
                        0 0 14px rgba(56, 216, 255, 0.10);
        }}

        .stButton > button[kind^="primary"]:hover,
        .stFormSubmitButton > button[kind^="primary"]:hover {{
            border-color: #b5f4ff;
            box-shadow: 0 0 0 1px rgba(56, 216, 255, 0.32),
                        0 0 22px rgba(56, 216, 255, 0.20);
            transform: translateY(-1px);
        }}

        .stButton > button:focus-visible,
        .stFormSubmitButton > button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="combobox"]:focus-visible,
        [role="tab"]:focus-visible {{
            outline: 3px solid rgba(56, 216, 255, 0.62) !important;
            outline-offset: 2px;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="slider"] {{
            background-color: rgba(21, 25, 31, 0.94);
            border-color: var(--q-border);
        }}

        button[role="tab"] {{
            color: var(--q-muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            font-weight: 750;
            transition: color 140ms ease, text-shadow 140ms ease;
        }}

        button[role="tab"][aria-selected="true"] {{
            color: #c8f7ff;
            text-shadow: 0 0 16px rgba(56, 216, 255, 0.20);
        }}

        .q-header {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.4rem 0 0.9rem;
            border-bottom: 1px solid var(--q-border);
            margin-bottom: 0.75rem;
        }}

        .q-title {{
            margin: 0;
            font-size: clamp(1.55rem, 3vw, 2.25rem);
            line-height: 1;
            letter-spacing: 0.12em;
            font-weight: 790;
            color: var(--q-text);
            background: linear-gradient(105deg, #ffffff 0%, #e9fbff 58%, #b9eaff 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 26px rgba(56, 216, 255, 0.075);
        }}

        .q-subtitle {{ color: #aab5c0; margin-top: 0.48rem; font-size: 0.92rem; }}

        .q-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            white-space: nowrap;
            color: #d2dbe3;
            background: linear-gradient(135deg, rgba(21, 25, 31, 0.94), rgba(13, 22, 29, 0.96));
            border: 1px solid #2b4651;
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            font: 650 0.72rem ui-monospace, SFMono-Regular, Consolas, monospace;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }}

        .q-status-dot {{
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: var(--q-cyan);
            box-shadow: 0 0 0 3px rgba(56, 216, 255, 0.08),
                        0 0 9px rgba(56, 216, 255, 0.48);
        }}

        .q-identity-strip {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.1rem 0 0.8rem;
            color: var(--q-muted);
            font: 700 0.69rem ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: 0.08em;
        }}

        .q-identity-token {{
            color: #d6f7ff;
            border: 1px solid #245160;
            background: linear-gradient(145deg, rgba(16, 37, 44, 0.92), rgba(14, 28, 35, 0.88));
            border-radius: 5px;
            padding: 0.28rem 0.45rem;
        }}

        .q-section-label {{
            margin: 0 0 0.55rem;
            color: #9ea9b5;
            font: 750 0.69rem ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .q-help {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.05rem;
            height: 1.05rem;
            margin-left: 0.32rem;
            color: #c7d0d8;
            border: 1px solid #4a5662;
            border-radius: 50%;
            cursor: help;
            font-size: 0.66rem;
            line-height: 1;
        }}

        .q-pipeline {{
            display: grid;
            grid-template-columns: repeat(7, minmax(88px, 1fr));
            gap: 0.36rem;
            margin: 0.35rem 0 0.75rem;
        }}

        .q-stage {{
            position: relative;
            min-height: 3.15rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 0.44rem;
            color: #737e89;
            background: rgba(14, 17, 21, 0.88);
            border: 1px solid #222933;
            border-radius: 7px;
            font: 700 0.62rem/1.28 ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: 0.035em;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.018);
            transition: border-color 140ms ease, background-color 140ms ease,
                        box-shadow 140ms ease, color 140ms ease;
        }}

        .q-stage.active {{
            color: #d9f9ff;
            border-color: #2d6c81;
            background: linear-gradient(155deg, rgba(15, 39, 48, 0.92), rgba(14, 28, 36, 0.88));
        }}

        .q-stage.active.classical:hover {{
            border-color: #398aa3;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035),
                        0 0 12px rgba(56, 216, 255, 0.07);
        }}

        .q-stage.quantum {{
            border-color: #5ccff1;
            background: linear-gradient(145deg, rgba(17, 44, 54, 0.96), rgba(29, 25, 54, 0.88));
            box-shadow: 0 0 0 1px rgba(56, 216, 255, 0.16),
                        0 0 14px rgba(92, 119, 255, 0.12);
        }}

        .q-stage.risk {{
            color: #ffd9c4;
            border-color: #9c5733;
            background: linear-gradient(155deg, rgba(42, 24, 17, 0.96), rgba(32, 20, 17, 0.92));
            box-shadow: 0 0 11px rgba(255, 138, 69, 0.09);
        }}

        .q-card {{
            background: linear-gradient(155deg, rgba(21, 25, 31, 0.94), rgba(16, 19, 24, 0.97));
            border: 1px solid var(--q-border);
            border-radius: 10px;
            padding: 0.95rem;
            margin-bottom: 0.75rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025),
                        0 8px 24px rgba(0, 0, 0, 0.11);
            transition: border-color 140ms ease, box-shadow 140ms ease,
                        transform 140ms ease;
        }}

        .q-card:hover {{ border-color: #364451; transform: translateY(-1px); }}

        .q-card.selected {{
            border-color: var(--q-cyan);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035),
                        0 0 0 1px rgba(56, 216, 255, 0.18),
                        0 0 16px rgba(56, 216, 255, 0.10);
        }}

        .q-card.risk {{
            border-color: var(--q-ember);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025),
                        0 0 0 1px rgba(255, 138, 69, 0.15),
                        0 0 14px rgba(255, 138, 69, 0.08);
        }}

        .q-eyebrow {{
            color: var(--q-muted);
            font: 750 0.65rem ui-monospace, SFMono-Regular, Consolas, monospace;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }}

        .q-route-id {{
            margin: 0.42rem 0 0.25rem;
            color: var(--q-text);
            font: 780 1.6rem ui-monospace, SFMono-Regular, Consolas, monospace;
        }}

        .q-badge {{
            display: inline-block;
            margin: 0.3rem 0 0.65rem;
            border: 1px solid #4b5966;
            border-radius: 999px;
            padding: 0.24rem 0.5rem;
            color: #d6dde4;
            font: 700 0.66rem ui-monospace, SFMono-Regular, Consolas, monospace;
        }}

        .q-badge.good {{ color: #d8f8ff; border-color: #2f788a; background: #10262d; }}
        .q-badge.risk {{ color: #ffe1cf; border-color: #a85d36; background: #2b1912; }}

        .q-kv {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.45rem 0.8rem;
            border-top: 1px solid #222a33;
            padding-top: 0.7rem;
            color: var(--q-muted);
            font-size: 0.76rem;
        }}

        .q-kv strong {{
            color: var(--q-text);
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-weight: 650;
        }}

        .q-empty {{
            min-height: 13rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1rem;
            color: var(--q-muted);
            background: linear-gradient(155deg, rgba(20, 24, 30, 0.78), rgba(15, 18, 23, 0.92));
            border: 1px dashed #35404b;
            border-radius: 10px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
        }}

        @media (max-width: 980px) {{
            .q-pipeline {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
            .q-header {{ align-items: flex-start; flex-direction: column; }}
            div[data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
                gap: 0.9rem !important;
            }}
            div[data-testid="stColumn"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
        }}

        @media (hover: none), (pointer: coarse) {{
            #q-pointer-glow {{ display: none !important; }}
            .q-card:hover,
            .stButton > button:hover,
            .stFormSubmitButton > button:hover {{ transform: none; }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            #q-pointer-glow {{ display: none !important; }}
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _install_pointer_glow()
