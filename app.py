"""
FIFA Match Predictor — Streamlit app.
Entry point: streamlit run app.py
"""

import os
import json
from html import escape

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="World Cup 2026 Match Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from data.ingest import load_all, get_all_teams, count_loaded
from data.world_cup_2026 import get_non_world_cup_2026_squad, is_qualified_team
from model.features import FEATURE_COLUMNS

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "artifacts")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
UI_BUILD = "2026.06.11-r9"


def format_probability_shift(value: float) -> str:
    """Format a probability shift without presenting rounded zero as positive."""
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "0.0%"
    return f"{percentage_points:+.1f}%"


def format_probability_point_note(value: float) -> str:
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "0.0 percentage points"
    return f"{percentage_points:+.1f} percentage points"


def shift_status(value: float, positive_text: str, negative_text: str) -> str:
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "No material change"
    return positive_text if percentage_points > 0 else negative_text


def inject_tournament_theme():
    """Apply the World Cup 2026-inspired visual system."""
    st.markdown(
        """
        <style>
        :root {
            --night: #06172b;
            --night-2: #0b2340;
            --surface: rgba(13, 41, 71, 0.86);
            --surface-soft: rgba(255, 255, 255, 0.055);
            --line: rgba(255, 255, 255, 0.13);
            --text: #f7f4ea;
            --muted: #a9bdd0;
            --cyan: #16c6e8;
            --blue: #2378ff;
            --lime: #b9f227;
            --yellow: #ffd229;
            --red: #ff3b4f;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            background:
                radial-gradient(circle at 11% 6%, rgba(35, 120, 255, 0.23), transparent 24rem),
                radial-gradient(circle at 88% 15%, rgba(255, 59, 79, 0.16), transparent 23rem),
                linear-gradient(180deg, #07192e 0%, #061526 52%, #04111f 100%);
            color: var(--text);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.26;
            background-image:
                linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: linear-gradient(to bottom, black, transparent 72%);
        }

        [data-testid="stHeader"] {
            background: rgba(4, 17, 31, 0.72);
            border-bottom: 1px solid rgba(255,255,255,.06);
            backdrop-filter: blur(18px);
        }

        [data-testid="stToolbar"] { right: 1.25rem; }

        .block-container {
            max-width: 1320px;
            padding-top: 3.25rem;
            padding-bottom: 5rem;
        }

        h1, h2, h3, [data-testid="stMetricValue"] {
            font-family: "Avenir Next Condensed", "DIN Condensed", "Arial Narrow", sans-serif !important;
            letter-spacing: -0.025em;
        }

        h1, h2, h3 {
            color: var(--text) !important;
        }

        p, label, li, button, input, [data-baseweb="select"] {
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }

        h2 {
            font-size: clamp(1.65rem, 3vw, 2.25rem) !important;
            margin-top: 0.45rem !important;
            padding-bottom: 0.55rem !important;
            border-bottom: 1px solid var(--line);
        }

        h2::before {
            content: "";
            display: inline-block;
            width: 0.42rem;
            height: 1.55rem;
            margin-right: 0.7rem;
            vertical-align: -0.16rem;
            border-radius: 99px;
            background: linear-gradient(var(--cyan), var(--lime));
            box-shadow: 0 0 22px rgba(22, 198, 232, .38);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        hr {
            border-color: var(--line) !important;
            margin: 2.25rem 0 !important;
        }

        .wc-hero {
            position: relative;
            overflow: hidden;
            min-height: 330px;
            padding: clamp(2rem, 5vw, 4.25rem);
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 28px;
            background:
                linear-gradient(115deg, rgba(6, 23, 43, .98) 10%, rgba(8, 40, 69, .86) 58%, rgba(12, 50, 75, .7)),
                repeating-linear-gradient(104deg, transparent 0 40px, rgba(255,255,255,.03) 40px 41px);
            box-shadow: 0 30px 80px rgba(0,0,0,.36);
        }

        .wc-hero::before {
            content: "26";
            position: absolute;
            right: -0.02em;
            bottom: -0.31em;
            font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
            font-size: clamp(15rem, 30vw, 29rem);
            font-weight: 900;
            letter-spacing: -0.12em;
            line-height: 1;
            color: transparent;
            -webkit-text-stroke: 1px rgba(255,255,255,.09);
            transform: skewX(-7deg);
        }

        .wc-hero::after {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 8px;
            background: linear-gradient(90deg,
                var(--blue) 0 20%, var(--cyan) 20% 40%, var(--lime) 40% 60%,
                var(--yellow) 60% 80%, var(--red) 80%);
        }

        .wc-kicker {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--lime);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .wc-kicker::before {
            content: "";
            width: 1.9rem;
            height: 2px;
            background: var(--lime);
        }

        .wc-hero h1 {
            position: relative;
            z-index: 1;
            max-width: 820px;
            margin: 1rem 0 .8rem;
            color: #fff;
            font-size: clamp(3.4rem, 8vw, 7.4rem);
            font-weight: 900;
            line-height: .83;
            letter-spacing: -.055em;
            text-transform: uppercase;
        }

        .wc-hero h1 span { color: var(--cyan); }

        .wc-hero-copy {
            position: relative;
            z-index: 1;
            max-width: 650px;
            margin: 0;
            color: #c5d7e7;
            font-size: clamp(.98rem, 1.7vw, 1.15rem);
            line-height: 1.7;
        }

        .wc-status-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            margin: 1rem 0 2.4rem;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--line);
        }

        .wc-status-item {
            padding: 1.05rem 1.2rem;
            background: rgba(7, 28, 50, .92);
        }

        .wc-status-label {
            display: block;
            margin-bottom: .3rem;
            color: #7f9bb4;
            font-size: .66rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .wc-status-value {
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
        }

        .wc-live-dot {
            display: inline-block;
            width: .52rem;
            height: .52rem;
            margin-right: .4rem;
            border-radius: 50%;
            background: var(--lime);
            box-shadow: 0 0 0 5px rgba(185,242,39,.09);
        }

        .wc-build-badge {
            position: absolute;
            z-index: 2;
            top: 1.6rem;
            right: 1.6rem;
            padding: .55rem .8rem;
            border: 1px solid rgba(185,242,39,.34);
            border-radius: 999px;
            background: rgba(4, 17, 31, .88);
            box-shadow: 0 10px 30px rgba(0,0,0,.3);
            color: var(--lime);
            font-size: .65rem;
            font-weight: 900;
            letter-spacing: .12em;
            text-transform: uppercase;
            backdrop-filter: blur(14px);
        }

        .wc-section-kicker {
            margin: 0 0 .25rem;
            color: var(--cyan);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .17em;
            text-transform: uppercase;
        }

        .wc-section-title {
            margin: 0 0 1rem;
            color: #fff;
            font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 900;
            letter-spacing: -.035em;
            line-height: 1;
            text-transform: uppercase;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 22px !important;
            background:
                linear-gradient(135deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
            box-shadow: 0 18px 55px rgba(0,0,0,.18);
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stRadio"] > label {
            color: #d7e5f0 !important;
            font-size: .68rem !important;
            font-weight: 800 !important;
            letter-spacing: .11em !important;
            text-transform: uppercase;
        }

        [data-baseweb="select"] > div {
            min-height: 3.7rem;
            padding: 0 .35rem;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,.22) !important;
            border-radius: 13px;
            background: #071b2f !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
        }

        [data-baseweb="select"] > div:hover {
            border-color: var(--cyan) !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        [data-baseweb="select"] > div > div {
            color: #fff !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        [data-baseweb="select"] svg {
            fill: var(--cyan) !important;
        }

        [data-baseweb="popover"] {
            z-index: 10000 !important;
        }

        [data-baseweb="popover"] > div {
            overflow: hidden !important;
            border: 1px solid rgba(22,198,232,.38) !important;
            border-radius: 14px !important;
            background: #071b2f !important;
            box-shadow: 0 24px 60px rgba(0,0,0,.48) !important;
        }

        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="popover"] ul,
        ul[role="listbox"],
        [role="listbox"] {
            max-height: min(22rem, 58vh) !important;
            padding: .4rem !important;
            background: #071b2f !important;
        }

        li[role="option"],
        [role="option"] {
            min-height: 2.7rem !important;
            margin: .12rem 0 !important;
            padding: .65rem .8rem !important;
            border-radius: 9px !important;
            background: #071b2f !important;
            color: #dce9f3 !important;
            font-size: .9rem !important;
            font-weight: 650 !important;
        }

        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background: rgba(22,198,232,.14) !important;
            color: #fff !important;
        }

        [role="option"][aria-selected="true"] {
            box-shadow: inset 3px 0 0 var(--lime);
        }

        .selector-note {
            margin: -.25rem 0 1rem;
            color: #8fa8bd;
            font-size: .7rem;
            line-height: 1.5;
        }

        [data-testid="stRadio"] [role="radiogroup"] {
            gap: .55rem;
        }

        [data-testid="stRadio"] [role="radiogroup"] label {
            min-height: 2.75rem;
            padding: .55rem .9rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255,255,255,.035);
        }

        .stButton > button {
            min-height: 3.45rem;
            border: 0;
            border-radius: 12px;
            background: linear-gradient(90deg, #12bfe2, #2378ff);
            box-shadow: 0 12px 30px rgba(35,120,255,.27);
            color: #fff;
            font-size: .84rem;
            font-weight: 900;
            letter-spacing: .13em;
            text-transform: uppercase;
            transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
        }

        .stButton > button:hover {
            border: 0;
            color: #fff;
            filter: brightness(1.08);
            transform: translateY(-2px);
            box-shadow: 0 17px 38px rgba(35,120,255,.35);
        }

        .action-spacer { height: 1.55rem; }

        .match-board {
            position: relative;
            overflow: hidden;
            padding: clamp(1.6rem, 4vw, 3rem);
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 24px;
            background:
                linear-gradient(90deg, rgba(255,255,255,.025) 50%, transparent 50%),
                repeating-linear-gradient(90deg, rgba(255,255,255,.025) 0 8%, transparent 8% 16%),
                linear-gradient(145deg, #087051, #075c43 72%);
            box-shadow: 0 24px 65px rgba(0,0,0,.28);
        }

        .match-board::before {
            content: "";
            position: absolute;
            inset: 1rem;
            opacity: .34;
            border: 2px solid #fff;
            border-radius: 4px;
            background:
                linear-gradient(90deg, transparent 49.8%, #fff 49.9% 50.1%, transparent 50.2%),
                radial-gradient(circle at 50% 50%, transparent 0 10%, #fff 10.2% 10.7%, transparent 10.9%);
        }

        .match-board::after {
            content: "";
            position: absolute;
            top: 28%;
            bottom: 18%;
            left: 1rem;
            right: 1rem;
            opacity: .3;
            background:
                linear-gradient(90deg,
                    transparent 0 12%,
                    #fff 12% 12.2%,
                    transparent 12.2% 87.8%,
                    #fff 87.8% 88%,
                    transparent 88%);
        }

        .match-meta {
            position: relative;
            z-index: 1;
            margin-bottom: 1.5rem;
            color: var(--lime);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-align: center;
            text-transform: uppercase;
        }

        .match-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center;
            gap: 1.5rem;
        }

        .team-card { min-width: 0; text-align: center; }

        .team-name {
            overflow-wrap: anywhere;
            color: #fff;
            font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
            font-size: clamp(2rem, 5vw, 4.6rem);
            font-weight: 900;
            letter-spacing: -.04em;
            line-height: .95;
            text-transform: uppercase;
        }

        .team-rank {
            display: inline-block;
            margin-top: .8rem;
            padding: .34rem .65rem;
            border: 1px solid rgba(255,255,255,.15);
            border-radius: 999px;
            color: var(--muted);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .ranking-source-note {
            position: relative;
            z-index: 1;
            margin-top: 1.35rem;
            color: rgba(255,255,255,.72);
            font-size: .64rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-align: center;
            text-transform: uppercase;
        }

        .versus {
            display: grid;
            width: 3.6rem;
            height: 3.6rem;
            place-items: center;
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 50%;
            background: rgba(3,17,31,.64);
            color: var(--yellow);
            font-family: "Avenir Next Condensed", sans-serif;
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: .08em;
        }

        .probability-wrap { margin: 1.4rem 0 1.8rem; }

        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-bottom: .55rem;
            color: var(--muted);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .probability-bar {
            display: flex;
            width: 100%;
            height: 44px;
            overflow: hidden;
            border: 3px solid rgba(255,255,255,.08);
            border-radius: 999px;
            background: #061526;
            box-shadow: 0 12px 30px rgba(0,0,0,.2);
        }

        .probability-bar > div {
            display: grid;
            min-width: 0;
            place-items: center;
            overflow: hidden;
            color: #fff;
            font-size: .76rem;
            font-weight: 900;
            white-space: nowrap;
            transition: width .65s cubic-bezier(.2,.8,.2,1);
        }

        .prob-a { background: linear-gradient(90deg, #1665e8, #16bde2); }
        .prob-draw { background: #53677b; }
        .prob-b { background: linear-gradient(90deg, #f04a58, #c8243b); }

        .report-section {
            margin: 1rem 0;
            padding: clamp(1.2rem, 2.8vw, 2rem);
            border: 1px solid var(--line);
            border-radius: 22px;
            background:
                linear-gradient(145deg, rgba(15, 43, 70, .9), rgba(7, 27, 47, .96));
            box-shadow: 0 18px 46px rgba(0,0,0,.18);
        }

        .report-heading {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.25rem;
            padding-bottom: .9rem;
            border-bottom: 1px solid var(--line);
        }

        .report-eyebrow {
            margin-bottom: .3rem;
            color: var(--lime);
            font-size: .63rem;
            font-weight: 900;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .report-title {
            color: #fff;
            font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
            font-weight: 900;
            letter-spacing: -.025em;
            line-height: 1;
            text-transform: uppercase;
        }

        .report-note {
            max-width: 44rem;
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.55;
            text-align: right;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }

        .stat-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .stat-grid.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }

        .stat-tile {
            min-width: 0;
            padding: 1.15rem;
            border: 1px solid rgba(255,255,255,.11);
            border-radius: 15px;
            background: rgba(255,255,255,.045);
        }

        .stat-tile.accent {
            background: linear-gradient(145deg, rgba(22,198,232,.14), rgba(35,120,255,.08));
        }

        .stat-label {
            display: block;
            min-height: 2.1em;
            margin-bottom: .55rem;
            color: var(--muted);
            font-size: .67rem;
            font-weight: 800;
            letter-spacing: .07em;
            line-height: 1.35;
            text-transform: uppercase;
        }

        .stat-value {
            display: block;
            color: #fff;
            font-family: "Avenir Next Condensed", "DIN Condensed", sans-serif;
            font-size: clamp(2rem, 4vw, 3.15rem);
            font-weight: 900;
            letter-spacing: -.035em;
            line-height: 1;
        }

        .stat-detail {
            display: block;
            margin-top: .55rem;
            color: #8fa8bd;
            font-size: .72rem;
            line-height: 1.4;
        }

        .shift-status {
            display: inline-flex;
            width: fit-content;
            margin-top: .8rem;
            padding: .3rem .55rem;
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 999px;
            color: #d7e5f0;
            background: rgba(255,255,255,.06);
            font-size: .62rem;
            font-weight: 900;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        .outcome-banner {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: .9rem;
            margin: 1rem 0;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(185,242,39,.28);
            border-radius: 15px;
            background: linear-gradient(90deg, rgba(8,112,81,.55), rgba(11,49,70,.88));
        }

        .outcome-icon { font-size: 1.35rem; }
        .outcome-label { color: var(--lime); font-size: .65rem; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }
        .outcome-value { color: #fff; font-size: 1.05rem; font-weight: 800; }
        .outcome-policy { color: var(--muted); font-size: .72rem; text-align: right; }

        .analysis-grid {
            display: grid;
            grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr);
            gap: 1rem;
        }

        .confidence-score {
            display: grid;
            min-height: 100%;
            padding: 1.4rem;
            place-content: center;
            border: 1px solid rgba(185,242,39,.22);
            border-radius: 16px;
            background:
                radial-gradient(circle at center, rgba(185,242,39,.12), transparent 64%),
                rgba(255,255,255,.035);
            text-align: center;
        }

        .confidence-score strong {
            color: #fff;
            font-family: "Avenir Next Condensed", sans-serif;
            font-size: clamp(3.2rem, 8vw, 5.5rem);
            line-height: 1;
        }

        .confidence-score span {
            margin-top: .5rem;
            color: var(--lime);
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .factor-list {
            display: grid;
            gap: .65rem;
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .factor-list li {
            display: grid;
            grid-template-columns: 1.7rem 1fr;
            align-items: start;
            gap: .65rem;
            padding: .8rem .9rem;
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 12px;
            background: rgba(255,255,255,.03);
            color: #dbe7f1;
            font-size: .83rem;
            line-height: 1.5;
        }

        .factor-index {
            display: grid;
            width: 1.7rem;
            height: 1.7rem;
            place-items: center;
            border-radius: 50%;
            background: rgba(22,198,232,.13);
            color: var(--cyan);
            font-size: .62rem;
            font-weight: 900;
        }

        .comparison-board {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 16px;
        }

        .comparison-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(9rem, .55fr) minmax(0, 1fr);
            align-items: center;
            min-height: 4.25rem;
            border-bottom: 1px solid var(--line);
        }

        .comparison-row:last-child { border-bottom: 0; }
        .comparison-row.header { min-height: 3.3rem; background: rgba(35,120,255,.14); }

        .comparison-cell {
            min-width: 0;
            padding: .8rem 1rem;
            color: #fff;
            font-size: .87rem;
            font-weight: 700;
        }

        .comparison-cell:nth-child(2) {
            border-right: 1px solid var(--line);
            border-left: 1px solid var(--line);
            color: var(--muted);
            font-size: .66rem;
            font-weight: 900;
            letter-spacing: .07em;
            text-align: center;
            text-transform: uppercase;
        }

        .comparison-cell:last-child { text-align: right; }
        .comparison-row.header .comparison-cell { color: #fff; font-size: .76rem; text-transform: uppercase; }
        .comparison-strong { color: var(--lime); }

        .verdict-card {
            padding: 1.25rem 1.4rem;
            border-left: 4px solid var(--cyan);
            border-radius: 4px 15px 15px 4px;
            background: rgba(22,198,232,.07);
            color: #e7f1f8;
            font-size: .95rem;
            line-height: 1.65;
        }

        .player-columns {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .player-team {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,.025);
        }

        .player-team-title {
            padding: .85rem 1rem;
            background: rgba(35,120,255,.13);
            color: #fff;
            font-size: .75rem;
            font-weight: 900;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        .player-card {
            padding: .9rem 1rem;
            border-top: 1px solid var(--line);
        }

        .player-name { color: #fff; font-weight: 800; }
        .player-meta { margin-top: .28rem; color: var(--muted); font-size: .73rem; line-height: 1.5; }
        .player-achievement { margin-top: .3rem; color: #8fa8bd; font-size: .72rem; line-height: 1.45; }

        [data-testid="stMetric"] {
            min-height: 122px;
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.025));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .07em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: #fff;
            font-size: clamp(1.9rem, 4vw, 3rem);
            font-weight: 900;
        }

        [data-testid="stAlert"] {
            border: 1px solid rgba(255,255,255,.13);
            border-radius: 14px;
            background: rgba(13, 41, 71, .88);
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] strong {
            color: var(--text) !important;
        }

        [data-testid="stMetricDelta"] {
            color: #8fa8bd !important;
        }

        [data-testid="stMetricDelta"] svg {
            fill: #8fa8bd !important;
        }

        [data-testid="stMarkdownContainer"] > p,
        [data-testid="stMarkdownContainer"] > ul,
        [data-testid="stMarkdownContainer"] > ol {
            color: #dbe7f1;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border-color: var(--line);
            border-radius: 16px;
            background: rgba(255,255,255,.035);
        }

        [data-testid="stExpander"] summary {
            min-height: 3.7rem;
            font-weight: 800;
        }

        table {
            overflow: hidden;
            border: 1px solid var(--line) !important;
            border-radius: 14px;
            background: rgba(4,20,36,.5);
        }

        table th {
            color: #fff !important;
            background: rgba(35,120,255,.16) !important;
        }

        table td, table th {
            border-color: var(--line) !important;
        }

        code {
            color: var(--lime) !important;
            background: rgba(185,242,39,.08) !important;
        }

        .wc-footer {
            margin-top: 3rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--line);
            color: #7892a9;
            font-size: .72rem;
            letter-spacing: .08em;
            text-align: center;
            text-transform: uppercase;
        }

        @media (max-width: 760px) {
            [data-testid="stHeader"] {
                height: 3.25rem;
                background: rgba(4, 17, 31, .94);
            }

            [data-testid="stToolbar"] {
                top: .35rem;
                right: .4rem;
            }

            .block-container {
                padding: 4rem .75rem 3rem;
            }

            .wc-hero {
                min-height: 0;
                padding: 1.35rem 1.15rem 2rem;
                border-radius: 18px;
            }

            .wc-build-badge {
                position: static;
                display: inline-flex;
                margin: .2rem 0 1.25rem;
                padding: .42rem .62rem;
                font-size: .55rem;
            }

            .wc-kicker {
                display: flex;
                gap: .5rem;
                font-size: .58rem;
                letter-spacing: .12em;
                line-height: 1.5;
            }

            .wc-kicker::before {
                width: 1.2rem;
                flex: 0 0 1.2rem;
            }

            .wc-hero h1 {
                margin: 1.05rem 0 1rem;
                font-size: clamp(2.7rem, 13vw, 4.2rem);
                line-height: .88;
                overflow-wrap: normal;
                word-break: normal;
            }

            .wc-hero h1 span {
                display: block;
                font-size: .72em;
                letter-spacing: -.045em;
                white-space: nowrap;
            }

            .wc-hero-copy {
                max-width: 100%;
                font-size: .92rem;
                line-height: 1.6;
            }

            .wc-status-grid {
                grid-template-columns: 1fr;
                margin-bottom: 1.8rem;
                border-radius: 14px;
            }

            .wc-status-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: .85rem 1rem;
            }

            .wc-status-label {
                margin: 0;
                font-size: .58rem;
            }

            .wc-status-value {
                max-width: 58%;
                font-size: .82rem;
                text-align: right;
                overflow-wrap: anywhere;
            }

            .wc-section-title {
                font-size: 2rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 16px !important;
            }

            [data-testid="stRadio"] [role="radiogroup"] {
                flex-wrap: wrap;
            }

            [data-baseweb="select"] > div {
                min-height: 3.4rem;
            }

            [data-baseweb="popover"] > div {
                max-width: calc(100vw - 1.5rem) !important;
            }

            [role="listbox"] {
                max-height: 19rem !important;
            }

            [role="option"] {
                min-height: 2.9rem !important;
                font-size: .92rem !important;
            }

            [data-testid="stRadio"] [role="radiogroup"] label {
                min-height: 2.45rem;
                padding: .45rem .65rem;
                font-size: .75rem;
            }

            .action-spacer { display: none; }

            .match-board {
                padding: 1.25rem .8rem;
                border-radius: 18px;
            }

            .match-board::before {
                inset: .55rem;
            }

            .match-board::after {
                left: .55rem;
                right: .55rem;
            }

            .match-meta {
                margin-bottom: 1.1rem;
                font-size: .58rem;
                letter-spacing: .1em;
            }

            .match-grid { gap: .45rem; }
            .team-name { font-size: clamp(1.25rem, 7vw, 2.1rem); }
            .versus { width: 2.7rem; height: 2.7rem; font-size: .75rem; }
            .team-rank { font-size: .56rem; }

            .probability-labels {
                font-size: .56rem;
                letter-spacing: .05em;
            }

            .probability-bar { height: 36px; }
            .probability-bar > div { font-size: .58rem; }

            [data-testid="stMetric"] {
                min-height: 100px;
                padding: .9rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.9rem;
            }

            h2 {
                font-size: 1.55rem !important;
            }

            table {
                font-size: .72rem !important;
            }

            .report-section {
                padding: 1rem;
                border-radius: 16px;
            }

            .report-heading {
                display: block;
                margin-bottom: 1rem;
            }

            .report-note {
                margin-top: .55rem;
                text-align: left;
            }

            .stat-grid {
                grid-template-columns: 1fr;
                gap: .7rem;
            }

            .stat-grid.two,
            .stat-grid.five {
                grid-template-columns: 1fr 1fr;
            }

            .stat-grid.five .stat-tile:first-child {
                grid-column: 1 / -1;
            }

            .stat-tile { padding: .9rem; }
            .stat-label { min-height: 0; font-size: .58rem; }
            .stat-value { font-size: 2rem; }

            .outcome-banner {
                grid-template-columns: auto 1fr;
                padding: .85rem;
            }

            .outcome-policy {
                grid-column: 2;
                text-align: left;
            }

            .analysis-grid,
            .player-columns {
                grid-template-columns: 1fr;
            }

            .comparison-row {
                grid-template-columns: minmax(0, 1fr) minmax(6.2rem, .7fr) minmax(0, 1fr);
                min-height: 3.7rem;
            }

            .comparison-cell {
                padding: .7rem .55rem;
                font-size: .72rem;
                overflow-wrap: anywhere;
            }

            .comparison-cell:nth-child(2) {
                font-size: .55rem;
            }

            [data-testid="stHorizontalBlock"] { gap: .7rem; }
        }

        @media (prefers-reduced-motion: no-preference) {
            .wc-hero, .wc-status-grid, [data-testid="stVerticalBlockBorderWrapper"] {
                animation: rise-in .65s ease both;
            }

            .wc-status-grid { animation-delay: .08s; }
            [data-testid="stVerticalBlockBorderWrapper"] { animation-delay: .15s; }

            @keyframes rise-in {
                from { opacity: 0; transform: translateY(14px); }
                to { opacity: 1; transform: translateY(0); }
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Model load-or-train (cached resource — runs once per session) ────────────

@st.cache_resource(show_spinner=False)
def get_model_and_meta(_artifact_stamp: tuple[float, float, float]):
    """Load model artifacts, training inline if not present."""
    from model.train import train_and_save

    if not os.path.exists(MODEL_PATH):
        st.info("No trained model found. Training now — this takes about 30 seconds...")
        meta = train_and_save(verbose=False)
    else:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        else:
            meta = {"model_name": "Unknown", "accuracy": 0.0, "training_rows": 0, "features": len(FEATURE_COLUMNS)}

    import joblib
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, meta


def _artifact_stamp() -> tuple[float, float, float]:
    def _mtime(path: str) -> float:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0

    return (_mtime(MODEL_PATH), _mtime(SCALER_PATH), _mtime(META_PATH))


# ── Data loading ──────────────────────────────────────────────────────────────

inject_tournament_theme()

with st.spinner("Loading match data..."):
    data = load_all()

matches_df = data["matches"]
goalscorers_df = data["goalscorers"]
shootouts_df = data["shootouts"]
rankings_df           = data["rankings"]
substitutions_df      = data.get("substitutions")
player_appearances_df = data.get("player_appearances")
player_goals_df       = data.get("player_goals")
award_winners_df      = data.get("award_winners")
copa_data             = data.get("copa_america")
euro_data             = data.get("euro_2024")
friendlies_data       = data.get("international_friendlies")
world_cup_2026_data   = data.get("world_cup_2026")

loaded_count = count_loaded(data)

if matches_df is None:
    st.error(
        "**matches.csv is required to run this app.**\n\n"
        "Drop it (and optionally goalscorers.csv, shootouts.csv, rankings.csv) "
        "into the project root directory, then refresh the page."
    )
    st.stop()

# ── Model ─────────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model_obj, scaler_obj, meta = get_model_and_meta(_artifact_stamp())

accuracy = meta.get("accuracy", 0.0)
training_rows = meta.get("training_rows", 0)
model_name = meta.get("model_name", "Unknown")

# ── Tournament masthead + status ──────────────────────────────────────────────

n_matches = len(matches_df)
st.markdown(
    f"""
    <section class="wc-hero">
        <div class="wc-build-badge">Live UI · Build {UI_BUILD}</div>
        <div class="wc-kicker">North America 2026 · Prediction Centre</div>
        <h1>Match<br><span>Intelligence</span></h1>
        <p class="wc-hero-copy">
            Explore international matchups through historical form, rankings,
            tournament pedigree, goal models, and 2026 squad intelligence.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="wc-status-grid">
        <div class="wc-status-item">
            <span class="wc-status-label">Historical matches</span>
            <span class="wc-status-value">{n_matches:,}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Prediction engine</span>
            <span class="wc-status-value"><span class="wc-live-dot"></span>{escape(str(model_name))}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Training rows</span>
            <span class="wc-status-value">{training_rows:,}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Core data coverage</span>
            <span class="wc-status-value">{loaded_count}/4 sources online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if accuracy < 0.55 and accuracy > 0:
    st.sidebar.warning(
        "Model accuracy is below 55%. Consider retraining with a narrower date range."
    )

# ── Matchup controls ──────────────────────────────────────────────────────────

all_teams = get_all_teams(matches_df)
ranked_teams = (
    set(rankings_df["country_full"].dropna().astype(str))
    if rankings_df is not None and "country_full" in rankings_df.columns
    else set()
)
qualified_teams = set()
if world_cup_2026_data:
    qualified_frame = world_cup_2026_data.get("teams")
    if qualified_frame is not None and not qualified_frame.empty:
        qualified_teams = set(qualified_frame["Team"].dropna().astype(str))

available_teams = set(all_teams)
selectable_teams = sorted(available_teams & qualified_teams) + sorted(
    (available_teams & ranked_teams) - qualified_teams
)
if not selectable_teams:
    selectable_teams = all_teams

st.markdown('<p class="wc-section-kicker">Build the fixture</p>', unsafe_allow_html=True)
st.markdown('<p class="wc-section-title">Choose your matchup</p>', unsafe_allow_html=True)
st.markdown(
    f"""
    <p class="selector-note">
        {len(selectable_teams)} active national teams · World Cup 2026 teams listed first · Type to search
    </p>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A",
            selectable_teams,
            index=selectable_teams.index("Argentina") if "Argentina" in selectable_teams else 0,
        )
    with col2:
        team_b = st.selectbox(
            "Team B",
            selectable_teams,
            index=selectable_teams.index("France") if "France" in selectable_teams else 1,
        )

    venue_col, action_col = st.columns([2, 1])
    with venue_col:
        venue_choice = st.radio(
            "Match setting",
            ["Team A home", "Neutral", "Team B home"],
            index=1,
            horizontal=True,
        )
    with action_col:
        st.markdown(
            '<div class="action-spacer" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        predict_btn = st.button(
            "Run match analysis",
            type="primary",
            use_container_width=True,
        )

venue_mode = {
    "Team A home": "team_a_home",
    "Neutral": "neutral",
    "Team B home": "team_b_home",
}[venue_choice]

# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    if team_a == team_b:
        st.error("Please select two different teams.")
        st.stop()

    with st.spinner(f"Analysing {n_matches:,} historical matches..."):
        from model.predict import predict_match
        result = predict_match(
            team_a=team_a,
            team_b=team_b,
            matches_df=matches_df,
            goalscorers_df=goalscorers_df,
            shootouts_df=shootouts_df,
            rankings_df=rankings_df,
            substitutions_df=substitutions_df,
            player_appearances_df=player_appearances_df,
            player_goals_df=player_goals_df,
            award_winners_df=award_winners_df,
            copa_data=copa_data,
            euro_data=euro_data,
            friendlies_data=friendlies_data,
            world_cup_2026_data=world_cup_2026_data,
            venue_mode=venue_mode,
        )

    win_prob = result["win_prob"]
    draw_prob = result["draw_prob"]
    loss_prob = result["loss_prob"]
    conf_score = result["confidence_score"]
    conf_label = result["confidence_label"]

    st.divider()

    # ── Match board ───────────────────────────────────────────────────────────
    rank_a = result["team_a_rank"]
    rank_b = result["team_b_rank"]
    rank_date_a = result.get("team_a_rank_date")
    rank_date_b = result.get("team_b_rank_date")
    rank_label_a = (
        pd.Timestamp(rank_date_a).strftime("%b %Y")
        if rank_date_a else "snapshot date unavailable"
    )
    rank_label_b = (
        pd.Timestamp(rank_date_b).strftime("%b %Y")
        if rank_date_b else "snapshot date unavailable"
    )
    rank_sources = {
        str(result.get("team_a_rank_source") or "Ranking snapshot"),
        str(result.get("team_b_rank_source") or "Ranking snapshot"),
    }
    display_rank_source = (
        "Official FIFA snapshot"
        if rank_sources == {"Official FIFA snapshot"}
        else "Latest available snapshot per team"
    )
    model_rank_date = result.get("model_rank_date")
    model_rank_label = (
        pd.Timestamp(model_rank_date).strftime("%b %Y")
        if model_rank_date else "date unavailable"
    )
    safe_team_a = escape(str(team_a))
    safe_team_b = escape(str(team_b))
    safe_venue = escape(str(result.get("venue_label", venue_choice)))
    st.markdown(
        f"""
        <section class="match-board">
            <div class="match-meta">World Cup 2026 Match Lab · {safe_venue}</div>
            <div class="match-grid">
                <div class="team-card">
                    <div class="team-name">{safe_team_a}</div>
                    <div class="team-rank">FIFA rank #{rank_a if rank_a else "N/A"} · {rank_label_a}</div>
                </div>
                <div class="versus">VS</div>
                <div class="team-card">
                    <div class="team-name">{safe_team_b}</div>
                    <div class="team-rank">FIFA rank #{rank_b if rank_b else "N/A"} · {rank_label_b}</div>
                </div>
            </div>
            <div class="ranking-source-note">
                Display ranking: {escape(display_rank_source)} · Model ranking features: {escape(model_rank_label)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Probabilities ─────────────────────────────────────────────────────────
    label_a = f"{team_a} ({win_prob:.0%})" if win_prob > 0.15 else f"{win_prob:.0%}" if win_prob > 0.05 else ""
    label_draw = f"Draw ({draw_prob:.0%})" if draw_prob > 0.15 else f"{draw_prob:.0%}" if draw_prob > 0.05 else ""
    label_b = f"{team_b} ({loss_prob:.0%})" if loss_prob > 0.15 else f"{loss_prob:.0%}" if loss_prob > 0.05 else ""

    html_prob_bar = f"""
    <div class="probability-wrap">
        <div class="probability-labels">
            <span>Outcome probability</span>
            <span>Model distribution · 100%</span>
        </div>
        <div class="probability-bar">
            <div class="prob-a" style="width:{win_prob*100:.2f}%">{escape(label_a)}</div>
            <div class="prob-draw" style="width:{draw_prob*100:.2f}%">{escape(label_draw)}</div>
            <div class="prob-b" style="width:{loss_prob*100:.2f}%">{escape(label_b)}</div>
        </div>
    </div>
    """
    st.markdown(html_prob_bar, unsafe_allow_html=True)

    predicted_label = result.get("predicted_label")
    if predicted_label == 2:
        outcome_text = f"{team_a} win"
        outcome_icon = "🏆"
    elif predicted_label == 0:
        outcome_text = f"{team_b} win"
        outcome_icon = "🏆"
    else:
        outcome_text = "Draw"
        outcome_icon = "🤝"

    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-tile accent">
                <span class="stat-label">{safe_team_a} win probability</span>
                <span class="stat-value">{win_prob:.1%}</span>
            </div>
            <div class="stat-tile">
                <span class="stat-label">Draw probability</span>
                <span class="stat-value">{draw_prob:.1%}</span>
            </div>
            <div class="stat-tile">
                <span class="stat-label">{safe_team_b} win probability</span>
                <span class="stat-value">{loss_prob:.1%}</span>
            </div>
        </div>
        <div class="outcome-banner">
            <div class="outcome-icon">{outcome_icon}</div>
            <div>
                <div class="outcome-label">Predicted outcome</div>
                <div class="outcome-value">{escape(outcome_text)}</div>
            </div>
            <div class="outcome-policy">{escape(result.get('decision_policy', 'highest probability'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2026 Squad Adjustments ────────────────────────────────────────────────
    wc2026_context = result.get("world_cup_2026_context", {})
    if wc2026_context.get("applied"):
        base = result.get("base_probabilities", {})
        shift_a = wc2026_context.get("team_a_probability_shift", 0)
        shift_draw = wc2026_context.get("draw_probability_shift", 0)
        shift_b = wc2026_context.get("team_b_probability_shift", 0)
        squad_edge_a = wc2026_context.get("team_a_squad_edge_shift", shift_a)
        squad_edge_b = wc2026_context.get("team_b_squad_edge_shift", shift_b)
        draw_context_shift = wc2026_context.get("draw_context_shift", shift_draw)
        base_a = float(base.get("win_prob", 0))
        base_draw = float(base.get("draw_prob", 0))
        base_b = float(base.get("loss_prob", 0))
        status_a = shift_status(squad_edge_a, "Squad edge boosts chance", "Squad edge reduces chance")
        status_draw = shift_status(draw_context_shift, "Close matchup boosts draw", "Close matchup lowers draw")
        status_b = shift_status(squad_edge_b, "Squad edge boosts chance", "Squad edge reduces chance")
        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Roster intelligence</div>
                        <div class="report-title">2026 squad impact</div>
                    </div>
                    <div class="report-note">
                        Probability movement after evaluating squad depth, age balance,
                        top-league experience, and the current 2026 roster snapshot.
                    </div>
                </div>
                <div class="stat-grid">
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_a} squad edge</span>
                        <span class="stat-value">{format_probability_shift(squad_edge_a)}</span>
                        <span class="shift-status">{escape(status_a)}</span>
                        <span class="stat-detail">Final win probability {base_a:.1%} → {win_prob:.1%} · net {format_probability_point_note(shift_a)}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Draw context shift</span>
                        <span class="stat-value">{format_probability_shift(draw_context_shift)}</span>
                        <span class="shift-status">{escape(status_draw)}</span>
                        <span class="stat-detail">Final draw probability {base_draw:.1%} → {draw_prob:.1%} · net {format_probability_point_note(shift_draw)}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_b} squad edge</span>
                        <span class="stat-value">{format_probability_shift(squad_edge_b)}</span>
                        <span class="shift-status">{escape(status_b)}</span>
                        <span class="stat-detail">Final win probability {base_b:.1%} → {loss_prob:.1%} · net {format_probability_point_note(shift_b)}</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # ── Goal prediction ──────────────────────────────────────────────────────
    if result.get("goals_error"):
        st.warning(f"Goal prediction is unavailable: {result['goals_error']}")
    elif "expected_team_a_goals" in result and "expected_team_b_goals" in result:
        goal_meta = result.get("goal_model_meta", {})
        goal_metrics = goal_meta.get("metrics", {})
        validation_note = (
            f"Validation: MAE {goal_metrics.get('combined_mae', 0):.2f} · "
            f"exact score {goal_metrics.get('exact_score_accuracy', 0):.1%} · "
            f"over 2.5 accuracy {goal_metrics.get('over_2_5_accuracy', 0):.1%}"
            if goal_metrics else "Independent expected-goals model"
        )
        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Score forecast</div>
                        <div class="report-title">Goal prediction</div>
                    </div>
                    <div class="report-note">
                        {safe_venue} · {escape(validation_note)}
                    </div>
                </div>
                <div class="stat-grid five">
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_a} expected goals</span>
                        <span class="stat-value">{result['expected_team_a_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile accent">
                        <span class="stat-label">Likely score</span>
                        <span class="stat-value">{result['likely_team_a_goals']}-{result['likely_team_b_goals']}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_b} expected goals</span>
                        <span class="stat-value">{result['expected_team_b_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Total expected goals</span>
                        <span class="stat-value">{result['expected_total_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Over 2.5 goals</span>
                        <span class="stat-value">{result['over_2_5_probability']:.1%}</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # ── Confidence ────────────────────────────────────────────────────────────
    n_used = result["n_features_used"]
    n_total = result["n_total"] if "n_total" in result else result.get("n_features_total", 10)
    factors_html = "".join(
        f'<li><span class="factor-index">{idx:02d}</span><span>{escape(str(factor))}</span></li>'
        for idx, factor in enumerate(result["key_factors"], start=1)
    )
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Model context</div>
                    <div class="report-title">Confidence & deciding factors</div>
                </div>
                <div class="report-note">
                    {n_used} of {n_total} data categories integrated · Rating: {escape(conf_label)}
                </div>
            </div>
            <div class="analysis-grid">
                <div class="confidence-score">
                    <strong>{conf_score:.0%}</strong>
                    <span>{escape(conf_label)} confidence</span>
                </div>
                <ul class="factor-list">{factors_html}</ul>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Historical stats ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Tournament pedigree</div>
                    <div class="report-title">World Cup history</div>
                </div>
                <div class="report-note">Side-by-side historical comparison</div>
            </div>
            <div class="comparison-board">
                <div class="comparison-row header">
                    <div class="comparison-cell">{safe_team_a}</div>
                    <div class="comparison-cell">Metric</div>
                    <div class="comparison-cell">{safe_team_b}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_avg_goals_wc']:.2f}</div>
                    <div class="comparison-cell">Goals per game</div>
                    <div class="comparison-cell">{result['team_b_avg_goals_wc']:.2f}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_wc_titles']}</div>
                    <div class="comparison-cell">World Cup titles</div>
                    <div class="comparison-cell">{result['team_b_wc_titles']}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_penalty_shootout_wins']}W / {result['team_a_penalty_shootout_losses']}L</div>
                    <div class="comparison-cell">Penalty shootouts</div>
                    <div class="comparison-cell">{result['team_b_penalty_shootout_wins']}W / {result['team_b_penalty_shootout_losses']}L</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_clean_sheet_rate']:.1%}</div>
                    <div class="comparison-cell">Clean sheet rate</div>
                    <div class="comparison-cell">{result['team_b_clean_sheet_rate']:.1%}</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Official 2026 squad stats ────────────────────────────────────────────
    squad_a = result.get("team_a_2026_squad", {})
    squad_b = result.get("team_b_2026_squad", {})
    non_wc_squad_a = get_non_world_cup_2026_squad(team_a, world_cup_2026_data)
    non_wc_squad_b = get_non_world_cup_2026_squad(team_b, world_cup_2026_data)
    if squad_a or squad_b or non_wc_squad_a or non_wc_squad_b:
        def _fallback_squad(team_name: str, fallback_squad: list[dict]):
            if not fallback_squad:
                return
            source = fallback_squad[0]
            st.caption(
                f"Latest 2026 non-World-Cup matchday squad for {team_name}: "
                f"{source.get('source_match', 'recent match')} ({source.get('source_date', 'date unavailable')})."
            )
            starters = [p for p in fallback_squad if p.get("squad_role") == "Starter"]
            substitutes = [p for p in fallback_squad if p.get("squad_role") != "Starter"]
            if starters:
                st.markdown(
                    "**Starters:** "
                    + ", ".join(f"#{p.get('shirt_number')} {p.get('player_name')}" for p in starters)
                )
            if substitutes:
                st.markdown(
                    "**Substitutes:** "
                    + ", ".join(f"#{p.get('shirt_number')} {p.get('player_name')}" for p in substitutes)
                )

        if squad_a and squad_b:
            html_table = f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Team composition</div>
                        <div class="report-title">2026 squad comparison</div>
                    </div>
                    <div class="report-note">Current roster profile and coaching context</div>
                </div>
                <div class="comparison-board">
                    <div class="comparison-row header">
                        <div class="comparison-cell">{safe_team_a}</div>
                        <div class="comparison-cell">Attribute</div>
                        <div class="comparison-cell">{safe_team_b}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('depth_score', 0):.2f}</div>
                        <div class="comparison-cell">Depth score</div>
                        <div class="comparison-cell">{squad_b.get('depth_score', 0):.2f}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('avg_age', 0):.1f} yrs</div>
                        <div class="comparison-cell">Average age</div>
                        <div class="comparison-cell">{squad_b.get('avg_age', 0):.1f} yrs</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('top_league_share', 0):.0%}</div>
                        <div class="comparison-cell">Top-league share</div>
                        <div class="comparison-cell">{squad_b.get('top_league_share', 0):.0%}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('avg_height_cm', 0):.1f} cm</div>
                        <div class="comparison-cell">Average height</div>
                        <div class="comparison-cell">{squad_b.get('avg_height_cm', 0):.1f} cm</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('goalkeepers', 0)} GK · {squad_a.get('defenders', 0)} DF · {squad_a.get('midfielders', 0)} MF · {squad_a.get('forwards', 0)} FW</div>
                        <div class="comparison-cell">Positions</div>
                        <div class="comparison-cell">{squad_b.get('goalkeepers', 0)} GK · {squad_b.get('defenders', 0)} DF · {squad_b.get('midfielders', 0)} MF · {squad_b.get('forwards', 0)} FW</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{escape(str(squad_a.get('head_coach', 'N/A')))}<br><span class="stat-detail">{escape(str(squad_a.get('coach_nationality', 'N/A')))}</span></div>
                        <div class="comparison-cell">Head coach</div>
                        <div class="comparison-cell">{escape(str(squad_b.get('head_coach', 'N/A')))}<br><span class="stat-detail">{escape(str(squad_b.get('coach_nationality', 'N/A')))}</span></div>
                    </div>
                </div>
            </section>
            """
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="report-eyebrow">Team composition</div><div class="report-title">2026 squad information</div>',
                unsafe_allow_html=True,
            )
            def _squad_metrics(team_name: str, squad: dict, fallback_squad: list[dict], col):
                with col:
                    st.markdown(f"**{team_name}**")
                    if not squad:
                        if is_qualified_team(team_name, world_cup_2026_data):
                            st.caption("Official 2026 squad data unavailable.")
                        else:
                            st.caption("Not a qualified 2026 World Cup team; no official World Cup squad exists.")
                            _fallback_squad(team_name, fallback_squad)
                        return
                    st.metric("Squad depth parameter", f"{squad.get('depth_score', 0):.2f}")
                    st.metric("Average age", f"{squad.get('avg_age', 0):.1f}")
                    st.metric("Top-league share", f"{squad.get('top_league_share', 0):.0%}")
                    st.caption(
                        f"{squad.get('players', 0)} players · "
                        f"{squad.get('goalkeepers', 0)} GK / {squad.get('defenders', 0)} DF / "
                        f"{squad.get('midfielders', 0)} MF / {squad.get('forwards', 0)} FW · "
                        f"avg height {squad.get('avg_height_cm', 0):.1f} cm"
                    )
                    coach = squad.get("head_coach")
                    if coach:
                        st.caption(f"Head coach: {coach} ({squad.get('coach_nationality', 'N/A')})")

            sq_col1, sq_col2 = st.columns(2)
            _squad_metrics(team_a, squad_a, non_wc_squad_a, sq_col1)
            _squad_metrics(team_b, squad_b, non_wc_squad_b, sq_col2)


    # ── Super-sub alerts ──────────────────────────────────────────────────────
    ss_a = result.get("supersub_a")
    ss_b = result.get("supersub_b")
    if ss_a or ss_b:
        supersub_cards = []
        for ss in [ss_a, ss_b]:
            if ss:
                supersub_cards.append(
                    f"""
                    <div class="stat-tile">
                        <span class="stat-label">⚡ {escape(str(ss['team']))} super-sub</span>
                        <span class="outcome-value">{escape(str(ss['name']))}</span>
                        <span class="stat-detail">
                            {ss['goals_as_sub']} goals in {ss['sub_appearances']} substitute appearances ·
                            {ss['goal_rate']:.2f} goals per appearance
                        </span>
                    </div>
                    """
                )
        st.markdown(
            f'<div class="stat-grid two">{"".join(supersub_cards)}</div>',
            unsafe_allow_html=True,
        )

    # ── Model verdict ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Final assessment</div>
                    <div class="report-title">Model verdict</div>
                </div>
            </div>
            <div class="verdict-card">{escape(str(result['verdict']))}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Key players ───────────────────────────────────────────────────────────
    kp_a = result.get("key_players_a", [])
    kp_b = result.get("key_players_b", [])
    if kp_a or kp_b:
        AWARD_BADGE = {
            "Golden Ball":  "🏆 Golden Ball",
            "Golden Boot":  "👟 Golden Boot",
            "Golden Glove": "🧤 Golden Glove",
        }

        def _player_cards(players: list) -> str:
            if not players:
                return '<div class="player-card"><div class="player-meta">No player data available.</div></div>'
            cards = []
            for p in players:
                badges = " · ".join(AWARD_BADGE.get(a, a) for a in p.get("awards", []))
                position = f" · {p.get('position')}" if p.get("position") else ""
                if p.get("source") == "Official 2026 squad":
                    primary_meta = (
                        f"#{p.get('shirt_number', 'N/A')} · {p.get('club', 'Club N/A')} · "
                        f"age {p.get('age', 'N/A')} · {p.get('height_cm', 'N/A')} cm"
                    )
                else:
                    assists = int(p.get("assists", 0) or 0)
                    assists_text = f" · {assists} assists" if assists else ""
                    primary_meta = (
                        f"{p.get('goals', 0)} goals{assists_text} · {p.get('appearances', 0)} appearances · "
                        f"{p.get('goal_rate', 0):.2f} goals/app"
                    )
                impact = (
                    f" · Impact {p['impact_score']:.2f}"
                    if p.get("impact_score") is not None else ""
                )
                achievement_parts = [
                    str(value) for value in [
                        p.get("euro_summary", ""),
                        p.get("achievement", ""),
                        " · ".join(p.get("achievements", [])[:4]),
                        badges,
                    ] if value
                ]
                achievement_html = (
                    f'<div class="player-achievement">{escape(" · ".join(achievement_parts))}</div>'
                    if achievement_parts else ""
                )
                cards.append(
                    '<div class="player-card">'
                    f'<div class="player-name">{escape(str(p.get("name", "Unknown")))}{escape(position)}</div>'
                    f'<div class="player-meta">{escape(primary_meta + impact)}</div>'
                    f'{achievement_html}'
                    '</div>'
                )
            return "".join(cards)

        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Players to watch</div>
                        <div class="report-title">Key players</div>
                    </div>
                    <div class="report-note">Roster role, club context, and model impact</div>
                </div>
                <div class="player-columns">
                    <div class="player-team">
                        <div class="player-team-title">{safe_team_a}</div>
                        {_player_cards(kp_a)}
                    </div>
                    <div class="player-team">
                        <div class="player-team-title">{safe_team_b}</div>
                        {_player_cards(kp_b)}
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="wc-footer">Independent prediction experience · Historical data and model estimates are informational</div>',
    unsafe_allow_html=True,
)
