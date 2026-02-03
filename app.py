"""
NBA Betting AI Dashboard v3 — Parlay-First Edition
Run: python app.py
"""
import dash
from dash import html, dcc, dash_table, callback_context, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import subprocess
import sys
from datetime import datetime

from config import *
from parlay_engine import build_optimal_parlays, format_parlay_summary

# ─── Initialize App ───────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="NBA Parlay AI",
    suppress_callback_exceptions=True,
)
server = app.server

# ─── Color Scheme ─────────────────────────────────────────────────────────────
C = {
    "bg": "#0f1117",
    "card": "#1a1d26",
    "card2": "#22252e",
    "border": "#2a2d36",
    "accent": "#6c63ff",
    "accent2": "#00d4aa",
    "green": "#00d4aa",
    "red": "#ff4757",
    "orange": "#ffa502",
    "gold": "#ffd700",
    "text": "#e8e8e8",
    "muted": "#8b8d97",
    "dim": "#555",
}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_predictions():
    try:
        return pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def load_model_info():
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return {}


def load_parlays(predictions_df):
    """Build parlays from current predictions"""
    if predictions_df is None or predictions_df.empty:
        return []
    try:
        return build_optimal_parlays(predictions_df, top_n=12)
    except Exception as e:
        print(f"Parlay build error: {e}")
        return []


# ─── Helper Components ────────────────────────────────────────────────────────

def ev_color(ev):
    if ev >= 10: return C["green"]
    elif ev >= 3: return C["accent2"]
    elif ev > 0: return C["orange"]
    return C["red"]


def grade_badge(grade, color, size="md"):
    sizes = {"sm": ("11px", "0px 6px"), "md": ("14px", "2px 10px"), "lg": ("18px", "4px 14px")}
    fs, pad = sizes.get(size, sizes["md"])
    return html.Span(grade, style={
        "fontSize": fs, "fontWeight": "800", "letterSpacing": "0.5px",
        "color": C["bg"], "background": color,
        "padding": pad, "borderRadius": "6px",
        "display": "inline-block", "textAlign": "center",
    })


def stat_pill(label, value, color=C["muted"]):
    return html.Div([
        html.Span(label, style={"fontSize": "10px", "color": C["muted"], "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        html.Div(value, style={"fontSize": "18px", "fontWeight": "700", "color": color}),
    ], style={"textAlign": "center", "padding": "8px 16px"})


def progress_ring(pct, size=80, color=C["green"]):
    """SVG circular progress indicator"""
    r = size / 2 - 6
    circ = 2 * 3.14159 * r
    offset = circ * (1 - pct / 100)
    return html.Div(
        html.Div(
            f"{pct:.0f}%",
            style={
                "position": "absolute", "top": "50%", "left": "50%",
                "transform": "translate(-50%, -50%)",
                "fontSize": f"{size//5}px", "fontWeight": "700", "color": color,
            }
        ),
        style={
            "width": f"{size}px", "height": f"{size}px",
            "borderRadius": "50%",
            "background": f"conic-gradient({color} {pct}%, {C['border']} {pct}%)",
            "position": "relative",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
        }
    )


# ─── Parlay Card ──────────────────────────────────────────────────────────────

def parlay_card(parlay, rank=1):
    """Render a single parlay recommendation card"""
    grade = parlay["grade"]
    color = parlay["color"]
    action = parlay["action"]
    num_legs = parlay["num_legs"]
    win_prob = parlay["win_prob"]
    combined_odds = parlay["combined_odds"]
    ev = parlay["ev"]
    kelly = parlay["kelly_pct"]
    is_sgp = parlay.get("is_sgp", False)
    legs = parlay["legs"]

    # American odds display
    if combined_odds >= 2.0:
        american_odds = f"+{int((combined_odds - 1) * 100)}"
    else:
        american_odds = f"-{int(100 / (combined_odds - 1))}"

    return dbc.Col(
        html.Div([
            # ── Header: Grade + Action + Rank ──
            html.Div([
                html.Div([
                    grade_badge(grade, color, "lg"),
                    html.Div([
                        html.Span(action, style={"fontSize": "12px", "fontWeight": "700", "color": color, "letterSpacing": "1px"}),
                        html.Span(f" #{rank}", style={"fontSize": "11px", "color": C["muted"], "marginLeft": "6px"}),
                    ], style={"marginLeft": "12px"}),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span(f"{num_legs} Legs", style={
                        "fontSize": "11px", "padding": "2px 8px", "borderRadius": "10px",
                        "background": C["accent"] + "33", "color": C["accent"], "fontWeight": "600",
                    }),
                    html.Span(" SGP", style={
                        "fontSize": "10px", "padding": "2px 6px", "borderRadius": "10px",
                        "background": C["orange"] + "33", "color": C["orange"], "fontWeight": "600",
                        "marginLeft": "6px",
                    }) if is_sgp else None,
                ]),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"}),

            # ── Key Metrics Row ──
            html.Div([
                # Win probability ring
                html.Div([
                    progress_ring(win_prob, 70, color),
                    html.Div("WIN RATE", style={"fontSize": "9px", "color": C["muted"], "textAlign": "center", "marginTop": "4px", "letterSpacing": "1px"}),
                ], style={"marginRight": "20px"}),

                # Stats
                html.Div([
                    html.Div([
                        html.Span("Payout ", style={"fontSize": "11px", "color": C["muted"]}),
                        html.Span(f"{combined_odds:.1f}x", style={"fontSize": "20px", "fontWeight": "700", "color": C["text"]}),
                        html.Span(f" ({american_odds})", style={"fontSize": "12px", "color": C["muted"], "marginLeft": "4px"}),
                    ]),
                    html.Div([
                        html.Span("EV ", style={"fontSize": "11px", "color": C["muted"]}),
                        html.Span(f"{ev:+.1f}%", style={"fontSize": "18px", "fontWeight": "700", "color": ev_color(ev)}),
                    ], style={"marginTop": "4px"}),
                    html.Div([
                        html.Span("Bet Size ", style={"fontSize": "11px", "color": C["muted"]}),
                        html.Span(f"{min(kelly, 5.0):.1f}%", style={"fontSize": "14px", "fontWeight": "600", "color": C["text"]}),
                        html.Span(" of bankroll", style={"fontSize": "10px", "color": C["dim"]}),
                    ], style={"marginTop": "4px"}),
                ], style={"flex": 1}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

            # ── Divider ──
            html.Hr(style={"borderColor": C["border"], "margin": "12px 0"}),

            # ── Legs ──
            html.Div([
                _parlay_leg(leg, i + 1, color) for i, leg in enumerate(legs)
            ]),

            # ── $100 Bet Example ──
            html.Div([
                html.Span("$100 bet ", style={"fontSize": "11px", "color": C["muted"]}),
                html.Span(f"wins ${(combined_odds * 100 - 100):.0f}", style={
                    "fontSize": "13px", "fontWeight": "700", "color": C["green"],
                }),
            ], style={
                "textAlign": "center", "marginTop": "12px", "padding": "8px",
                "background": C["green"] + "0D", "borderRadius": "8px",
            }),

        ], style={
            "background": C["card"],
            "border": f"1px solid {color}44",
            "borderRadius": "16px",
            "padding": "20px",
            "transition": "transform 0.2s ease",
        }),
        xs=12, sm=12, md=6, lg=4, className="mb-4",
    )


def _parlay_leg(leg, num, parlay_color):
    """Render a single leg inside a parlay card"""
    conf = leg["confidence"]
    odds = leg["odds"]

    # Confidence bar color
    if conf >= 70: bar_color = C["green"]
    elif conf >= 60: bar_color = C["accent"]
    else: bar_color = C["orange"]

    return html.Div([
        html.Div([
            # Leg number
            html.Span(f"LEG {num}", style={
                "fontSize": "9px", "fontWeight": "700", "color": C["muted"],
                "letterSpacing": "1px", "marginRight": "8px",
            }),
            # Bet type badge
            html.Span(leg["bet_type"], style={
                "fontSize": "9px", "padding": "1px 6px", "borderRadius": "4px",
                "background": C["accent"] + "22", "color": C["accent"], "fontWeight": "600",
            }),
        ], style={"marginBottom": "4px"}),

        # Pick
        html.Div([
            html.Span(leg["pick"], style={"fontWeight": "700", "fontSize": "14px", "color": C["text"]}),
        ], style={"marginBottom": "2px"}),

        # Game + details
        html.Div([
            html.Span(f"{leg['home_team']} vs {leg['away_team']}", style={"fontSize": "11px", "color": C["muted"]}),
            html.Span(f" · {leg['game_time']}", style={"fontSize": "10px", "color": C["dim"]}),
        ], style={"marginBottom": "6px"}),

        # Confidence bar + stats
        html.Div([
            html.Div([
                html.Div(style={
                    "width": f"{min(conf, 100)}%", "height": "4px",
                    "background": bar_color, "borderRadius": "2px",
                })
            ], style={
                "flex": 1, "height": "4px", "background": C["border"],
                "borderRadius": "2px", "overflow": "hidden", "marginRight": "10px",
            }),
            html.Span(f"{conf:.0f}%", style={"fontSize": "12px", "fontWeight": "600", "color": bar_color}),
            html.Span(f" · {odds:.2f}", style={"fontSize": "11px", "color": C["muted"]}),
            html.Span(f" @ {leg.get('book', 'N/A')}", style={"fontSize": "10px", "color": C["dim"]}) if leg.get("book", "N/A") != "N/A" else None,
        ], style={"display": "flex", "alignItems": "center"}),

    ], style={
        "padding": "10px 12px",
        "borderLeft": f"3px solid {parlay_color}55",
        "marginBottom": "8px",
        "borderRadius": "4px",
        "background": C["card2"],
    })


# ─── Individual Game Card ─────────────────────────────────────────────────────

def game_card(row):
    """Single game prediction card (enhanced for parlay view)"""
    ht = row["home_team"]
    at = row["away_team"]
    ml_pick = row["ml_pick"]
    ml_conf = row["ml_confidence"]
    ml_ev = row["ml_ev"]
    ml_odds = row["ml_odds"]
    home_prob = row.get("ml_home_prob", 50)
    away_prob = row.get("ml_away_prob", 50)
    game_time = row.get("game_time", "TBD")

    # Check if any bet qualifies for parlays
    parlay_ready = (
        (ml_conf >= PARLAY_MIN_CONFIDENCE and ml_ev >= PARLAY_MIN_EV) or
        (row.get("total_confidence", 0) >= PARLAY_MIN_CONFIDENCE and row.get("total_ev", 0) >= PARLAY_MIN_EV) or
        (row.get("spread_confidence", 0) >= PARLAY_MIN_CONFIDENCE and row.get("spread_ev", 0) >= PARLAY_MIN_EV)
    )

    # Grade the best individual bet
    bets = [
        ("ML", ml_pick, ml_conf, ml_ev, ml_odds, row.get("ml_book", "")),
        ("O/U", f"{row.get('total_pick', 'N/A')} {row.get('total_line', '')}", row.get("total_confidence", 50), row.get("total_ev", 0), row.get("total_odds", 1.9), row.get("total_book", "")),
        ("SPR", row.get("spread_pick", "N/A"), row.get("spread_confidence", 50), row.get("spread_ev", 0), row.get("spread_odds", 1.9), row.get("spread_book", "")),
    ]

    return dbc.Col(
        html.Div([
            # Header
            html.Div([
                html.Span(game_time, style={"color": C["muted"], "fontSize": "12px"}),
                html.Span("PARLAY READY", style={
                    "fontSize": "9px", "padding": "2px 8px", "borderRadius": "10px",
                    "background": C["green"] + "22", "color": C["green"], "fontWeight": "700",
                    "letterSpacing": "0.5px",
                }) if parlay_ready else None,
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"}),

            # Teams + Probability
            html.Div([
                html.Div([
                    html.Span(ht, style={
                        "fontWeight": "700" if ml_pick == ht else "400",
                        "color": C["accent"] if ml_pick == ht else C["text"],
                        "fontSize": "15px",
                    }),
                    html.Span(f" {home_prob}%", style={"color": C["muted"], "fontSize": "12px", "marginLeft": "6px"}),
                ]),
                html.Div([
                    html.Div(style={"width": f"{home_prob}%", "height": "6px", "background": C["accent"], "borderRadius": "3px 0 0 3px"}),
                    html.Div(style={"width": f"{away_prob}%", "height": "6px", "background": C["red"], "borderRadius": "0 3px 3px 0"}),
                ], style={"display": "flex", "margin": "4px 0", "borderRadius": "3px", "overflow": "hidden"}),
                html.Div([
                    html.Span(at, style={
                        "fontWeight": "700" if ml_pick == at else "400",
                        "color": C["red"] if ml_pick == at else C["text"],
                        "fontSize": "15px",
                    }),
                    html.Span(f" {away_prob}%", style={"color": C["muted"], "fontSize": "12px", "marginLeft": "6px"}),
                ]),
            ], style={"marginBottom": "14px"}),

            html.Hr(style={"borderColor": C["border"], "margin": "10px 0"}),

            # Bet rows
            html.Div([
                _simple_bet_row(label, pick, conf, ev, odds, book) for label, pick, conf, ev, odds, book in bets
            ]),

        ], style={
            "background": C["card"],
            "border": f"1px solid {C['green']}44" if parlay_ready else f"1px solid {C['border']}",
            "borderRadius": "16px",
            "padding": "18px",
        }),
        xs=12, sm=6, md=4, lg=3, className="mb-4",
    )


def _simple_bet_row(label, pick, conf, ev, odds, book):
    color = ev_color(ev)
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "10px", "color": C["muted"], "fontWeight": "600", "width": "32px", "display": "inline-block"}),
            html.Span(pick, style={"fontSize": "12px", "color": C["text"], "fontWeight": "600"}),
        ]),
        html.Div([
            html.Span(f"{conf:.0f}%", style={"fontSize": "11px", "color": C["text"], "marginRight": "8px"}),
            html.Span(f"{ev:+.1f}%", style={"fontSize": "11px", "color": color, "fontWeight": "600"}),
        ]),
    ], style={"display": "flex", "justifyContent": "space-between", "padding": "4px 0"})


# ─── Layout ───────────────────────────────────────────────────────────────────

def serve_layout():
    df = load_predictions()
    parlays = load_parlays(df)

    # Stats
    n_games = len(df) if not df.empty else 0
    n_parlays = len(parlays)
    model_acc = "N/A"
    brier = "N/A"
    try:
        pkg = load_model_info()
        model_acc = f"{pkg.get('calibrated_accuracy', pkg.get('cv_accuracy', 0)) * 100:.1f}%"
        brier = f"{pkg.get('brier_score', 0):.3f}"
    except Exception:
        pass

    best_parlay_win = f"{parlays[0]['win_prob']:.0f}%" if parlays else "N/A"

    return dbc.Container([
        # ── Header ──
        html.Div([
            html.Div([
                html.H1("NBA Parlay AI", style={"fontWeight": "800", "margin": 0, "fontSize": "28px"}),
                html.P("Calibrated ML predictions optimized for parlays", style={"color": C["muted"], "margin": 0, "fontSize": "13px"}),
            ]),
            html.Div([
                dbc.Button("Refresh", id="btn-refresh", color="primary", size="sm", className="me-2",
                           style={"borderRadius": "8px", "fontWeight": "600"}),
                dbc.Button("Retrain", id="btn-retrain", color="secondary", size="sm",
                           style={"borderRadius": "8px", "fontWeight": "600"}),
            ]),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "padding": "20px 0", "borderBottom": f"1px solid {C['border']}",
            "marginBottom": "16px",
        }),

        # Status
        html.Div(id="status-msg"),
        dcc.Loading(id="loading", type="circle", children=[html.Div(id="loading-output")]),

        # ── Stats Row ──
        dbc.Row([
            dbc.Col(html.Div([
                html.P("GAMES", style={"color": C["muted"], "fontSize": "10px", "margin": 0, "letterSpacing": "1px"}),
                html.H3(str(n_games), style={"color": C["accent"], "margin": "2px 0", "fontWeight": "700"}),
            ], style={"background": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "12px 16px", "textAlign": "center"}), xs=6, md=3, className="mb-2"),
            dbc.Col(html.Div([
                html.P("PARLAYS", style={"color": C["muted"], "fontSize": "10px", "margin": 0, "letterSpacing": "1px"}),
                html.H3(str(n_parlays), style={"color": C["green"], "margin": "2px 0", "fontWeight": "700"}),
            ], style={"background": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "12px 16px", "textAlign": "center"}), xs=6, md=3, className="mb-2"),
            dbc.Col(html.Div([
                html.P("ACCURACY", style={"color": C["muted"], "fontSize": "10px", "margin": 0, "letterSpacing": "1px"}),
                html.H3(model_acc, style={"color": C["accent2"], "margin": "2px 0", "fontWeight": "700"}),
            ], style={"background": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "12px 16px", "textAlign": "center"}), xs=6, md=3, className="mb-2"),
            dbc.Col(html.Div([
                html.P("BEST WIN%", style={"color": C["muted"], "fontSize": "10px", "margin": 0, "letterSpacing": "1px"}),
                html.H3(best_parlay_win, style={"color": C["gold"], "margin": "2px 0", "fontWeight": "700"}),
            ], style={"background": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "12px 16px", "textAlign": "center"}), xs=6, md=3, className="mb-2"),
        ], className="mb-3"),

        # ── Tabs ──
        dbc.Tabs([
            dbc.Tab(label="Parlays", tab_id="tab-parlays"),
            dbc.Tab(label="All Picks", tab_id="tab-all"),
            dbc.Tab(label="Analytics", tab_id="tab-analytics"),
            dbc.Tab(label="Guide", tab_id="tab-guide"),
        ], id="tabs", active_tab="tab-parlays", className="mb-3",
            style={"borderBottom": f"1px solid {C['border']}"}),

        html.Div(id="tab-content"),

        # Footer
        html.P(
            f"Updated: {datetime.now().strftime('%b %d, %Y %I:%M %p')} · Calibrated XGBoost + Isotonic · Not financial advice",
            style={"color": C["dim"], "fontSize": "10px", "textAlign": "center", "margin": "30px 0 10px"},
        ),

    ], fluid=True, style={
        "background": C["bg"], "minHeight": "100vh",
        "padding": "0 24px", "maxWidth": "1400px", "color": C["text"],
    })


app.layout = serve_layout


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab(tab):
    df = load_predictions()

    if tab == "tab-parlays":
        return render_parlays(df)
    elif tab == "tab-all":
        return render_all_picks(df)
    elif tab == "tab-analytics":
        return render_analytics(df)
    elif tab == "tab-guide":
        return render_guide()
    return html.Div()


def render_parlays(df):
    """Parlay recommendations tab — the main view"""
    if df.empty:
        return _empty_state("No predictions yet. Click 'Refresh' to generate picks.")

    parlays = load_parlays(df)

    if not parlays:
        return _empty_state("No parlays available. Need at least 2 games with high-confidence picks.")

    # Split by grade
    grade_a = [p for p in parlays if p["grade"] == "A"]
    grade_b = [p for p in parlays if p["grade"] == "B"]
    others = [p for p in parlays if p["grade"] not in ("A", "B")]

    sections = []

    if grade_a:
        sections.append(html.H4("Top Parlays", style={"color": C["green"], "fontWeight": "700", "margin": "16px 0 12px"}))
        sections.append(dbc.Row([parlay_card(p, i + 1) for i, p in enumerate(grade_a[:6])]))

    if grade_b:
        sections.append(html.H4("Solid Options", style={"color": C["accent2"], "fontWeight": "700", "margin": "20px 0 12px"}))
        sections.append(dbc.Row([parlay_card(p, len(grade_a) + i + 1) for i, p in enumerate(grade_b[:6])]))

    if others:
        sections.append(html.H4("Other Parlays", style={"color": C["muted"], "fontWeight": "700", "margin": "20px 0 12px"}))
        sections.append(dbc.Row([parlay_card(p, len(grade_a) + len(grade_b) + i + 1) for i, p in enumerate(others[:4])]))

    # Parlay math explainer
    sections.append(html.Div([
        html.Hr(style={"borderColor": C["border"], "margin": "24px 0"}),
        html.P([
            "Parlays multiply each leg's odds for bigger payouts but require ALL legs to win. ",
            "We only recommend parlays where each leg has ",
            html.Strong(f"{PARLAY_MIN_CONFIDENCE:.0f}%+ confidence", style={"color": C["green"]}),
            " and positive EV. Max ", html.Strong(f"{PARLAY_MAX_LEGS} legs", style={"color": C["accent"]}),
            " to keep win rates viable.",
        ], style={"color": C["muted"], "fontSize": "12px", "textAlign": "center"}),
    ]))

    return html.Div(sections)


def render_all_picks(df):
    """Individual game picks"""
    if df.empty:
        return _empty_state("No predictions yet. Click 'Refresh' to generate picks.")

    return dbc.Row([game_card(row) for _, row in df.iterrows()])


def render_analytics(df):
    """Analytics charts"""
    if df.empty:
        return _empty_state("No data for analytics.")

    charts = []

    # 1. EV by game
    fig_ev = go.Figure()
    labels = (df["home_team"].str.split().str[-1] + " v " + df["away_team"].str.split().str[-1])
    fig_ev.add_trace(go.Bar(
        x=labels, y=df["ml_ev"],
        marker_color=[ev_color(v) for v in df["ml_ev"]],
        text=[f"{v:+.1f}%" for v in df["ml_ev"]],
        textposition="outside",
    ))
    fig_ev.update_layout(
        title="Moneyline EV by Game", template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        font={"color": C["text"]}, height=320,
        margin=dict(l=40, r=20, t=50, b=60),
    )
    charts.append(dbc.Col(dcc.Graph(figure=fig_ev), xs=12, lg=6, className="mb-4"))

    # 2. Confidence distribution
    fig_conf = go.Figure()
    for label, col, color in [
        ("Moneyline", "ml_confidence", C["accent"]),
        ("Totals", "total_confidence", C["accent2"]),
        ("Spread", "spread_confidence", C["orange"]),
    ]:
        if col in df.columns:
            fig_conf.add_trace(go.Box(y=df[col], name=label, marker_color=color))
    fig_conf.update_layout(
        title="Confidence by Bet Type", template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        font={"color": C["text"]}, height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    charts.append(dbc.Col(dcc.Graph(figure=fig_conf), xs=12, lg=6, className="mb-4"))

    # 3. Model vs Market
    if "ml_model_home_prob" in df.columns and "ml_market_home_prob" in df.columns:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df["ml_market_home_prob"], y=df["ml_model_home_prob"],
            mode="markers+text",
            text=df["home_team"].str.split().str[-1],
            textposition="top center",
            textfont=dict(size=10, color=C["muted"]),
            marker=dict(size=12, color=df["ml_ev"], colorscale="RdYlGn", showscale=True, colorbar=dict(title="EV%")),
        ))
        fig_sc.add_trace(go.Scatter(x=[20, 80], y=[20, 80], mode="lines", line=dict(dash="dash", color=C["dim"]), showlegend=False))
        fig_sc.update_layout(
            title="Model vs Market Probability",
            xaxis_title="Market %", yaxis_title="Model %",
            template="plotly_dark", paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
            font={"color": C["text"]}, height=380,
            margin=dict(l=50, r=20, t=50, b=50),
        )
        charts.append(dbc.Col(dcc.Graph(figure=fig_sc), xs=12, className="mb-4"))

    return dbc.Row(charts)


def render_guide():
    """Parlay-focused guide"""
    S = {
        "section": {"background": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "14px", "padding": "24px 28px", "marginBottom": "20px"},
        "h2": {"color": C["text"], "fontWeight": "700", "fontSize": "20px", "marginBottom": "14px"},
        "h3": {"color": C["accent"], "fontWeight": "600", "fontSize": "15px", "marginTop": "16px", "marginBottom": "8px"},
        "p": {"color": C["muted"], "fontSize": "13px", "lineHeight": "1.7", "marginBottom": "10px"},
        "strong": {"color": C["text"], "fontWeight": "600"},
        "box": {"background": C["bg"], "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "14px 18px", "marginTop": "8px", "marginBottom": "12px", "fontFamily": "monospace", "fontSize": "12px", "color": C["text"], "lineHeight": "1.8"},
    }

    return html.Div([
        # Parlay Basics
        html.Div([
            html.H2("What is a Parlay?", style=S["h2"]),
            html.P([
                "A parlay combines multiple bets (called ", html.Strong("legs", style=S["strong"]),
                ") into one. ", html.Strong("ALL legs must win", style=S["strong"]),
                " for the parlay to pay out. The upside? Much bigger payouts.",
            ], style=S["p"]),
            html.Div([
                "2-Leg Parlay Example:", html.Br(),
                "  Leg 1: Clippers ML (1.78 odds) — 74% confidence", html.Br(),
                "  Leg 2: Under 231.5 (1.95 odds) — 71% confidence", html.Br(), html.Br(),
                "Combined odds: 1.78 x 1.95 = 3.47x", html.Br(),
                "Win probability: 74% x 71% = 52.5%", html.Br(),
                "$100 bet wins $247 profit", html.Br(), html.Br(),
                "Compare: $100 on each separately wins ~$78 + $95 = $173 profit",
            ], style=S["box"]),
        ], style=S["section"]),

        # Our Approach
        html.Div([
            html.H2("Our Parlay Strategy", style=S["h2"]),
            html.P("We use calibrated machine learning to build parlays that actually hit:", style=S["p"]),
            html.H3("1. Calibrated Probabilities", style=S["h3"]),
            html.P([
                "Our model uses ", html.Strong("isotonic calibration", style=S["strong"]),
                " — when it says 65% win probability, the pick actually wins ~65% of the time. ",
                "This is critical for parlay math because bad probabilities multiply into bad parlays.",
            ], style=S["p"]),
            html.H3("2. High Win Rate Focus", style=S["h3"]),
            html.P([
                "Every parlay leg must have at least ",
                html.Strong(f"{PARLAY_MIN_CONFIDENCE:.0f}% confidence", style={"color": C["green"], "fontWeight": "600"}),
                " and positive expected value. No filler legs.",
            ], style=S["p"]),
            html.H3(f"3. Maximum {PARLAY_MAX_LEGS} Legs", style=S["h3"]),
            html.P("More legs = exponentially lower win rate. We cap at 3 legs to keep parlays viable:", style=S["p"]),
            html.Div([
                "2 legs at 65% each = 42% parlay win rate  (GOOD)", html.Br(),
                "3 legs at 65% each = 27% parlay win rate  (OK)", html.Br(),
                "4 legs at 65% each = 18% parlay win rate  (RISKY)", html.Br(),
                "5 legs at 65% each = 12% parlay win rate  (LOTTERY)",
            ], style=S["box"]),
            html.H3("4. Correlation Awareness", style=S["h3"]),
            html.P("We check if parlay legs are correlated (same division, same game). Correlated legs reduce true win probability, so we adjust accordingly.", style=S["p"]),
        ], style=S["section"]),

        # Grade System
        html.Div([
            html.H2("Parlay Grades", style=S["h2"]),
            html.P("Each parlay gets a grade based on: 50% win probability + 30% EV + 20% consistency", style=S["p"]),
            _grade_row("A", "STRONG PARLAY", C["green"], "2-leg: 42%+ win rate | 3-leg: 28%+ win rate | Strong EV"),
            _grade_row("B", "SOLID PARLAY", C["accent2"], "2-leg: 36%+ win rate | 3-leg: 22%+ win rate | Good EV"),
            _grade_row("C+", "LEAN PARLAY", C["orange"], "Meets minimum thresholds. Lower confidence."),
            _grade_row("D", "SKIP", C["red"], "Negative EV or below thresholds. Don't bet."),
        ], style=S["section"]),

        # Bankroll
        html.Div([
            html.H2("Bankroll Management", style=S["h2"]),
            html.P("For parlays, bet conservatively:", style=S["p"]),
            html.Div([
                "Grade A parlay: 2-3% of bankroll", html.Br(),
                "Grade B parlay: 1-2% of bankroll", html.Br(),
                "Grade C+: 0.5-1% of bankroll (or skip)", html.Br(),
                "Grade D: $0 (skip entirely)", html.Br(), html.Br(),
                "NEVER exceed 5% on any single parlay.", html.Br(),
                "NEVER chase losses with bigger parlays.",
            ], style=S["box"]),
        ], style=S["section"]),

        # Glossary
        html.Div([
            html.H2("Quick Glossary", style=S["h2"]),
            _gloss("Parlay", "Multi-leg bet. All legs must win. Higher payout, lower probability."),
            _gloss("Leg", "A single pick within a parlay."),
            _gloss("SGP", "Same-Game Parlay. Multiple bets from one game. Slightly correlated."),
            _gloss("EV", "Expected Value. Positive = profitable long-term."),
            _gloss("Calibration", "When model probabilities match real-world outcomes."),
            _gloss("Kelly %", "Optimal bet size formula. We use quarter Kelly for safety."),
            _gloss("Moneyline", "Bet on which team wins. No point spread."),
            _gloss("Spread", "Handicap bet. Favorite must win by X+ points."),
            _gloss("Over/Under", "Bet on total combined points vs a set line."),
            _gloss("Decimal Odds", "Total return per $1 bet. 2.50 = $2.50 back ($1.50 profit)."),
        ], style=S["section"]),
    ])


def _grade_row(grade, label, color, desc):
    return html.Div([
        html.Div([
            grade_badge(grade, color),
            html.Span(label, style={"fontSize": "12px", "fontWeight": "700", "color": color, "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.P(desc, style={"color": C["muted"], "fontSize": "12px", "marginLeft": "42px", "marginBottom": "12px"}),
    ], style={"padding": "8px 0", "borderBottom": f"1px solid {C['border']}"})


def _gloss(term, defn):
    return html.Div([
        html.Span(term, style={"color": C["accent"], "fontWeight": "600", "fontSize": "13px", "minWidth": "140px", "display": "inline-block"}),
        html.Span(f" — {defn}", style={"color": C["muted"], "fontSize": "12px"}),
    ], style={"padding": "6px 0", "borderBottom": f"1px solid {C['border']}"})


def _empty_state(msg):
    return html.Div([
        html.H4("No Data", style={"textAlign": "center", "color": C["muted"], "marginTop": "60px"}),
        html.P(msg, style={"textAlign": "center", "color": C["dim"]}),
    ])


# ─── Button Callbacks ─────────────────────────────────────────────────────────

@app.callback(
    [Output("status-msg", "children"), Output("loading-output", "children")],
    [Input("btn-refresh", "n_clicks"), Input("btn-retrain", "n_clicks")],
    prevent_initial_call=True,
)
def handle_buttons(refresh_clicks, retrain_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return "", ""

    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    venv_python = os.path.join(DATA_DIR, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    if btn == "btn-retrain":
        try:
            result = subprocess.run(
                [venv_python, os.path.join(DATA_DIR, "model_engine.py")],
                capture_output=True, text=True, timeout=600, cwd=DATA_DIR,
            )
            if result.returncode == 0:
                msg = dbc.Alert("Model retrained! Refresh page to see updated picks.", color="success", dismissable=True)
            else:
                msg = dbc.Alert(f"Training error: {result.stderr[-300:]}", color="warning", dismissable=True)
        except subprocess.TimeoutExpired:
            msg = dbc.Alert("Training timed out (>10 min).", color="warning", dismissable=True)
        except Exception as e:
            msg = dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
        return msg, ""

    elif btn == "btn-refresh":
        try:
            result = subprocess.run(
                [venv_python, "-c", """
import sys; sys.path.insert(0, '.')
from data_pipeline import fetch_live_odds, parse_odds, load_historical_data, build_features
from model_engine import load_models, predict_games
odds = fetch_live_odds()
if odds:
    odds_df = parse_odds(odds)
    df = load_historical_data()
    features = build_features(df)
    pkg = load_models()
    predict_games(odds_df, features, pkg)
    print("OK")
else:
    print("NO_ODDS")
"""],
                capture_output=True, text=True, timeout=300, cwd=DATA_DIR,
            )
            if "OK" in result.stdout:
                msg = dbc.Alert("Predictions refreshed! Reload the page.", color="success", dismissable=True)
            elif "NO_ODDS" in result.stdout:
                msg = dbc.Alert("No odds available (no games today or off-season).", color="info", dismissable=True)
            else:
                msg = dbc.Alert("No trained model found. Click 'Retrain' first.", color="warning", dismissable=True)
        except Exception as e:
            msg = dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
        return msg, ""

    return "", ""


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NBA Parlay AI Dashboard v3")
    print("  http://127.0.0.1:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
