"""The Fitting Room: interactive results for "Generalization in Deep Networks" on Fashion-MNIST.

Author: Anush Goel

Run from the folder that contains the notebook's `artifacts/` directory:
    streamlit run app.py

Everything shown here is read from `artifacts/`. Predictions in the "Try it on" view are computed in NumPy
from the exported weights, so TensorFlow is not needed to run the app.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fitting_room import artifacts as fr_artifacts  # noqa: E402
from fitting_room.analysis import forward, mcnemar_exact, perturb, relative_signal, simulate_early_stopping  # noqa: E402
from fitting_room.hypotheses import INCONCLUSIVE, SUPPORTED  # noqa: E402
from fitting_room.hypotheses import evaluate as evaluate_hypotheses  # noqa: E402
from fitting_room.report import build_report  # noqa: E402

st.set_page_config(page_title="The Fitting Room", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------------------------
BG = "#131B31"        # deep indigo denim
PANEL = "#1B2543"     # raised denim
RAISED = "#26335C"
THREAD = "#E8A33D"    # gold topstitching, used for UI accents only
INK = "#E9ECF5"
MUTED = "#9AA3BD"
GRID = "rgba(233, 236, 245, 0.08)"
KRAFT, KRAFT_INK = "#D8B98C", "#2A2116"

MODEL_COLORS = {  # notebook palette, lifted slightly for a dark background
    "baseline": "#A3ABC2", "he_init": "#6C95F0", "l2_0.001": "#3CCB9F", "l2_0.01": "#1F9C86",
    "dropout_0.2": "#F2906E", "dropout_0.4": "#D9506B", "tuned": "#A88BF0",
}
CLASS_COLORS = ["#6C95F0", "#F2906E", "#3CCB9F", "#E8C35A", "#D9506B", "#5FC8E8", "#C48BF0", "#9FD36A", "#F0A3C8", "#B8BCC9"]
CLASS_SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea {{
  font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ background: {BG}; color: {INK}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: 'Bricolage Grotesque', 'Atkinson Hyperlegible', system-ui, sans-serif;
  color: {INK}; letter-spacing: -0.012em;
}}
.stApp h1 {{ font-weight: 800; font-size: clamp(2.4rem, 4.6vw, 3.6rem); line-height: 1.02; margin: 0.4rem 0 0.6rem; }}
.stApp h2 {{ font-weight: 700; font-size: 1.7rem; margin-top: 0.2rem; }}
.stApp h3 {{ font-weight: 600; font-size: 1.22rem; }}
[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px dashed rgba(232, 163, 61, 0.35); }}
.lede {{ font-size: 1.14rem; line-height: 1.6; color: #CBD1E2; max-width: 64ch; }}
.note {{ font-size: 0.94rem; line-height: 1.55; color: {MUTED}; max-width: 78ch; }}
.seam {{ border: 0; border-top: 2px dashed rgba(232, 163, 61, 0.5); margin: 2rem 0 1.4rem; }}
.brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.5rem; color: {INK}; margin: 0; }}
.brand-sub {{ color: {MUTED}; font-size: 0.9rem; margin: 0.2rem 0 1rem; }}
.tag {{
  position: relative; background: {KRAFT}; color: {KRAFT_INK};
  clip-path: polygon(13% 0, 100% 0, 100% 100%, 13% 100%, 0 50%);
  padding: 1.3rem 1.4rem 1.1rem 3.1rem; margin-top: 0.6rem;
  outline: 1.5px dashed rgba(42, 33, 22, 0.45); outline-offset: -9px;
}}
.tag-hole {{ position: absolute; left: 7.5%; top: 50%; width: 15px; height: 15px; margin-top: -7.5px;
  border-radius: 50%; background: {BG}; box-shadow: inset 0 0 0 2px rgba(42, 33, 22, 0.35); }}
.tag p {{ margin: 0; color: {KRAFT_INK}; }}
.tag .tag-title {{ font-size: 0.92rem; opacity: 0.8; }}
.tag .tag-model {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.45rem; line-height: 1.15; margin: 0.15rem 0 0.8rem; }}
.tag dl {{ margin: 0; display: grid; grid-template-columns: auto auto; gap: 0.25rem 1.2rem; }}
.tag dt {{ font-size: 0.92rem; opacity: 0.85; }}
.tag dd {{ margin: 0; font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }}
.tag .tag-foot {{ font-size: 0.82rem; opacity: 0.78; margin-top: 0.8rem; max-width: 34ch; }}
.pill {{ display: inline-block; padding: 0.1rem 0.6rem; border: 1px dashed rgba(232, 163, 61, 0.6); border-radius: 999px;
  color: {INK}; font-size: 0.88rem; margin: 0 0.35rem 0.35rem 0; }}
a {{ color: {THREAD}; }}
:focus-visible {{ outline: 2px solid {THREAD} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)


def seam():
    st.markdown('<hr class="seam">', unsafe_allow_html=True)


def lede(text):
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


def style(fig, height=430, legend=True, title=None):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Atkinson Hyperlegible, system-ui, sans-serif", color=INK, size=13),
        margin=dict(l=8, r=8, t=56 if title else 30, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=RAISED, font_color=INK, bordercolor=THREAD),
        title=dict(text=title, x=0, xanchor="left", font=dict(family="Bricolage Grotesque, sans-serif", size=17)) if title else None,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def fmt_pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
DEFAULT_ART = Path(__file__).resolve().parent / "artifacts"


@st.cache_resource(show_spinner="Loading results")
def load_artifacts(path_str):
    return fr_artifacts.load(path_str)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def garment(vec, scale=4, dark=PANEL, light="#F3E6CC"):
    """Render a 784-vector (0..1) as a denim-tinted RGB image."""
    v = np.clip(np.asarray(vec, dtype=np.float32).reshape(28, 28), 0, 1) ** 0.85
    rgb = hex_rgb(dark)[None, None, :] * (1 - v[..., None]) + hex_rgb(light)[None, None, :] * v[..., None]
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    return img.resize((28 * scale, 28 * scale), Image.NEAREST)


def mosaic(vecs, cols, scale=3, gap=4, bg=BG, **kw):
    n = len(vecs)
    rows = max(1, math.ceil(n / cols))
    tile = 28 * scale
    canvas = Image.new("RGB", (cols * tile + (cols - 1) * gap, rows * tile + (rows - 1) * gap), tuple(hex_rgb(bg).astype(int)))
    for i, v in enumerate(vecs):
        r, c = divmod(i, cols)
        canvas.paste(garment(v, scale, **kw), (c * (tile + gap), r * (tile + gap)))
    return canvas


# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
PAGES = ["Overview", "The garments", "Learning curves", "Initialization", "Regularization", "Early stopping",
         "Tuning", "Final model", "Head to head", "Hypotheses", "Run registry", "Try it on"]

with st.sidebar:
    st.markdown('<p class="brand">The Fitting Room</p>'
                '<p class="brand-sub">Generalization in deep networks, tried on Fashion-MNIST</p>', unsafe_allow_html=True)
    page = st.radio("View", PAGES, index=0, label_visibility="collapsed")
    with st.expander("Results folder"):
        art_path = st.text_input("Path to artifacts", value=str(DEFAULT_ART))
    st.markdown('<p class="note" style="margin-top:1.4rem">Built by Anush Goel</p>', unsafe_allow_html=True)
    if (st.get_option("theme.base") or "light") != "dark":
        st.caption("For the intended colors, keep the .streamlit folder next to app.py.")

A = load_artifacts(art_path)
R, F = A["runs"], A["final"]
if R is None or F is None:
    st.title("The Fitting Room")
    st.warning(f"No results found in {art_path}. Run the notebook through section 10 (Exporting results), "
               "make sure the artifacts folder sits next to app.py, then reload this page.")
    st.stop()

MAIN = [k for k in ["baseline", "he_init", "l2_0.001", "l2_0.01", "dropout_0.2", "dropout_0.4"] if k in R]
ALL = MAIN + (["tuned"] if "tuned" in R else [])
EPOCHS = len(R[MAIN[0]]["history"]["val_accuracy"])
SEL = F["selected"]
NAMES = F["class_names"]


def color(k):
    return MODEL_COLORS.get(k, R[k].get("color", THREAD))


def label(k):
    return R[k]["label"]


def short(k):
    return R[k]["short"]


# ----------------------------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------------------------
def view_overview():
    if A["preds"] is not None:
        Xv, yv = A["X_val"], A["preds"]["y_val"]
        picks = [int(np.where(yv == c)[0][i]) for i in range(3) for c in range(10)][:30]
        st.image(mosaic([Xv[i] for i in picks], cols=15, scale=3, gap=6))
    st.title("The fitting room")
    lede(f"One network, {len(MAIN)} ways to train it. Every model below is the same 784–128–128–10 ReLU network trained for "
         f"{EPOCHS} epochs with Adam; only the initializer or the regularizer changes. This is how each one fits the 50,000 "
         "training garments, and how much of that fit carries over to 10,000 garments it never trained on.")

    left, right = st.columns([2.3, 1], gap="large")
    with left:
        show(trajectory_figure(MAIN))
        note("Each path is one model, epoch by epoch. Horizontal position is training accuracy measured with dropout off; "
             "vertical position is validation accuracy. The dashed seam is where the two are equal. A path that keeps moving "
             "right without rising is fitting the training set more tightly without getting any better on new garments.")
    with right:
        lo, hi = F["ci_bootstrap"]
        st.markdown(f"""
<div class="tag"><div class="tag-hole"></div>
<p class="tag-title">Selected fit</p>
<p class="tag-model">{F['label']}</p>
<dl>
<dt>Validation accuracy</dt><dd>{fmt_pct(F['best_val_acc'])}</dd>
<dt>Test accuracy</dt><dd>{fmt_pct(F['test_acc'])}</dd>
<dt>95% interval</dt><dd>{100 * lo:.1f} to {100 * hi:.1f}%</dd>
<dt>Test loss</dt><dd>{F['test_loss']:.4f}</dd>
<dt>Best epoch</dt><dd>{F['best_epoch']} of {F['epochs']}</dd>
</dl>
<p class="tag-foot">Chosen on validation data only, from He initialization, the better L2 model, and the better dropout model.
The test set was used once.</p>
</div>""", unsafe_allow_html=True)

    seam()
    st.subheader("Table 1. Summary comparison")
    if A["table1"] is not None:
        st.dataframe(A["table1"], hide_index=True)
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button("Download Table 1 (CSV)", A["table1"].to_csv(index=False), "table1_summary.csv", "text/csv")
        with d2:
            st.download_button("Download results report", build_report(A, fig_rel="figures"), "RESULTS.md", "text/markdown")
    note("Training and validation accuracy are read at each model's epoch of best validation accuracy. For dropout models the "
         "training figure is measured with dropout active, which is why it can fall below validation accuracy.")

    seam()
    st.subheader("Fit versus generalization at the best epoch")
    fig = go.Figure()
    xs = [short(k) for k in ALL]
    clean = [R[k]["summary"]["clean_train_acc_at_best"] for k in ALL]
    val = [R[k]["summary"]["best_val_acc"] for k in ALL]
    fig.add_bar(x=xs, y=clean, name="Training, dropout off", marker=dict(color=[color(k) for k in ALL], opacity=0.45,
                                                                           line=dict(color=[color(k) for k in ALL], width=1.5)))
    fig.add_bar(x=xs, y=val, name="Validation", marker=dict(color=[color(k) for k in ALL]))
    for x, c_, v in zip(xs, clean, val):
        fig.add_annotation(x=x, y=max(c_, v), text=f"gap {100 * (c_ - v):.1f} pp", showarrow=False, yshift=14,
                           font=dict(color=MUTED, size=12))
    lo_y = min(min(clean), min(val)) - 0.02
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[lo_y, min(1.0, max(clean) + 0.02)], tickformat=".0%")
    show(style(fig, 380))


def trajectory_figure(keys):
    xs = {k: np.array(R[k]["diag"]["clean_train_acc"][1:]) for k in keys}
    ys = {k: np.array(R[k]["history"]["val_accuracy"]) for k in keys}
    allx, ally = np.concatenate(list(xs.values())), np.concatenate(list(ys.values()))
    x_lo, x_hi = float(allx.min()) - 0.006, float(min(1.0, allx.max() + 0.006))
    y_lo, y_hi = float(ally.min()) - 0.006, float(ally.max()) + 0.006
    seam_line = go.Scatter(x=[min(x_lo, y_lo), 1], y=[min(x_lo, y_lo), 1], mode="lines", name="training = validation",
                           line=dict(color=THREAD, dash="dash", width=1.4), hoverinfo="skip")

    def traces(t):
        out = [seam_line]
        for k in keys:
            out.append(go.Scatter(x=xs[k][:t], y=ys[k][:t], mode="lines", line=dict(color=color(k), width=2.4),
                                  legendgroup=k, showlegend=False, hoverinfo="skip"))
            out.append(go.Scatter(x=[xs[k][t - 1]], y=[ys[k][t - 1]], mode="markers", name=short(k), legendgroup=k,
                                  marker=dict(size=14, color=color(k), line=dict(color=BG, width=2)),
                                  hovertemplate=f"{label(k)}<br>epoch {t}<br>training %{{x:.2%}}<br>validation %{{y:.2%}}<extra></extra>"))
        return out

    frames = [go.Frame(data=traces(t), name=str(t)) for t in range(1, EPOCHS + 1)]
    fig = go.Figure(data=traces(EPOCHS), frames=frames)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=-0.16, xanchor="left", yanchor="top",
                          bgcolor=RAISED, bordercolor=THREAD, font=dict(color=INK),
                          buttons=[dict(label="Play from epoch 1", method="animate",
                                        args=[[str(t) for t in range(1, EPOCHS + 1)],
                                              {"frame": {"duration": 110, "redraw": False}, "transition": {"duration": 0},
                                               "mode": "immediate", "fromcurrent": False}]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        sliders=[dict(active=EPOCHS - 1, x=0.3, len=0.7, y=-0.1, yanchor="top", bgcolor=RAISED, activebgcolor=THREAD,
                      bordercolor=GRID, tickcolor=MUTED, font=dict(color=MUTED),
                      currentvalue=dict(prefix="Epoch ", font=dict(color=INK, size=14)),
                      steps=[dict(label=str(t), method="animate",
                                  args=[[str(t)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                             for t in range(1, EPOCHS + 1)])],
    )
    fig.update_xaxes(range=[x_lo, x_hi], tickformat=".0%", title="Training accuracy (dropout off)")
    fig.update_yaxes(range=[y_lo, y_hi], tickformat=".0%", title="Validation accuracy")
    fig.add_annotation(x=x_hi, y=y_lo, xanchor="right", yanchor="bottom", showarrow=False,
                       text="further right of the seam: training fit that does not transfer", font=dict(color=MUTED, size=12))
    style(fig, 560)
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=110))
    return fig


def view_garments():
    st.title("The garments")
    lede("Fashion-MNIST has ten balanced classes of 28 × 28 grayscale clothing images. Several upper-body classes look alike, "
         "which sets a ceiling on how well any model can do.")
    if A["preds"] is None or A["stats"] is None:
        st.info("Image data is missing from the artifacts folder.")
        return
    Xv, yv, S = A["X_val"], A["preds"]["y_val"], A["stats"]
    c = st.selectbox("Class", list(range(10)), format_func=lambda i: NAMES[i])
    idx = np.where(yv == c)[0][:36]
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        st.image(mosaic([Xv[i] for i in idx], cols=12, scale=3, gap=5))
        note(f"First 36 validation images labeled {NAMES[c]}.")
    with right:
        st.image(garment(S["mean_imgs"][c], scale=7))
        note(f"Average {NAMES[c]}.")
        sims = np.array(S["class_sim"][c], dtype=float); sims[c] = -np.inf
        order = np.argsort(-sims)[:3]
        st.markdown("Closest look-alikes by average shape: " + " ".join(
            f'<span class="pill">{NAMES[o]} {S["class_sim"][c][o]:.2f}</span>' for o in order), unsafe_allow_html=True)
        pc = {r["class"]: r for r in F["per_class"]}
        if NAMES[c] in pc:
            r = pc[NAMES[c]]
            note(f"Final model on the test set: recall {fmt_pct(r['Recall'], 1)}, precision {fmt_pct(r['Precision'], 1)}, F1 {r['F1']:.3f}.")
    seam()
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure(go.Heatmap(z=np.asarray(S["pixel_std"]).reshape(28, 28), colorscale=[[0, BG], [0.5, "#3E6FA8"], [1, "#BDE3F2"]],
                                   colorbar=dict(title="SD", tickfont=dict(color=MUTED)), hovertemplate="row %{y}, col %{x}<br>SD %{z:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False, showticklabels=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        show(style(fig, 420, legend=False, title="Where pixels actually vary"))
        flat = int((np.asarray(S["pixel_std"]) < 0.05).sum())
        note(f"{flat} of 784 pixels barely change across the training set (SD below 0.05). A dense layer still gives each of them 128 weights.")
    with b:
        fig = go.Figure(go.Heatmap(z=S["class_sim"], x=CLASS_SHORT, y=CLASS_SHORT, zmin=-1, zmax=1,
                                   colorscale=[[0, "#3C5BA9"], [0.5, PANEL], [1, THREAD]], text=np.round(S["class_sim"], 2),
                                   texttemplate="%{text}", textfont=dict(size=10), hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 420, legend=False, title="How alike the average garments are"))
        note("Cosine similarity of class-mean images after removing the overall mean image.")


METRICS = {
    "Accuracy": "acc", "Loss as Keras reports it": "loss", "Validation cross-entropy without the L2 term": "val_ce",
    "Generalization gap (dropout off)": "gap", "Training accuracy with dropout off": "clean", "Squared weight norm of hidden layers": "norm",
}


def series(k, metric):
    h, d = R[k]["history"], R[k]["diag"]
    ep = np.arange(1, EPOCHS + 1)
    if metric == "acc":
        return [(ep, h["val_accuracy"], "validation", "solid"), (ep, h["accuracy"], "training (Keras)", "dash")]
    if metric == "loss":
        return [(ep, h["val_loss"], "validation", "solid"), (ep, h["loss"], "training", "dash")]
    if metric == "val_ce":
        return [(np.arange(0, EPOCHS + 1), d["val_ce"], "validation CE", "solid")]
    if metric == "gap":
        return [(ep, np.array(d["clean_train_acc"][1:]) - np.array(h["val_accuracy"]), "gap", "solid")]
    if metric == "clean":
        return [(np.arange(0, EPOCHS + 1), d["clean_train_acc"], "training, dropout off", "solid")]
    return [(np.arange(0, EPOCHS + 1), np.array(d["sqnorm_h1"]) + np.array(d["sqnorm_h2"]), "ΣW²", "solid")]


def view_curves():
    st.title("Learning curves")
    lede("Pick the models and the quantity to compare. Stars mark each model's epoch of best validation accuracy.")
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    with c1:
        keys = st.multiselect("Models", ALL, default=MAIN, format_func=label)
    with c2:
        metric_name = st.selectbox("Quantity", list(METRICS))
    with c3:
        show_train = st.toggle("Show training curves", value=True)
    metric = METRICS[metric_name]
    ep_range = st.slider("Epochs", 0, EPOCHS, (0 if metric in ("val_ce", "clean", "norm") else 1, EPOCHS))
    if not keys:
        st.info("Choose at least one model.")
        return
    fig = go.Figure()
    for k in keys:
        for x, y, name, dash in series(k, metric):
            if dash == "dash" and not show_train:
                continue
            x, y = np.asarray(x), np.asarray(y, dtype=float)
            m = (x >= ep_range[0]) & (x <= ep_range[1])
            fig.add_scatter(x=x[m], y=y[m], mode="lines", name=f"{short(k)}, {name}", legendgroup=k,
                            line=dict(color=color(k), width=2.6 if dash == "solid" else 1.6, dash=dash),
                            hovertemplate=f"{short(k)} {name}<br>epoch %{{x}}<br>%{{y:.4f}}<extra></extra>")
        if metric == "acc":
            b = R[k]["summary"]["best_epoch"]
            if ep_range[0] <= b <= ep_range[1]:
                fig.add_scatter(x=[b], y=[R[k]["summary"]["best_val_acc"]], mode="markers", showlegend=False, legendgroup=k,
                                marker=dict(symbol="star", size=17, color=color(k), line=dict(color=BG, width=1.5)),
                                hovertemplate=f"{short(k)} best validation %{{y:.2%}} at epoch {b}<extra></extra>")
    if metric in ("acc", "clean"):
        fig.update_yaxes(tickformat=".0%")
    if metric == "gap":
        fig.update_yaxes(tickformat=".1%"); fig.add_hline(y=0, line_color=MUTED, line_width=1)
    if metric == "norm":
        fig.update_yaxes(type="log")
    fig.update_xaxes(title="Epoch")
    show(style(fig, 520))
    if metric == "loss" and any(R[k]["config"].get("l2") for k in keys):
        note("The Keras loss of the L2 models includes the penalty λΣW², so it is not comparable with the other models. "
             "Switch to the cross-entropy without the L2 term for a fair comparison.")
    rows = []
    for k in keys:
        s = R[k]["summary"]
        rows.append({"Model": label(k), "Best val. accuracy": fmt_pct(s["best_val_acc"]), "Best epoch": s["best_epoch"],
                     "Training accuracy at best (Keras)": fmt_pct(s["train_acc_at_best"]),
                     "Training accuracy at best (dropout off)": fmt_pct(s["clean_train_acc_at_best"]),
                     "Clean gap": f"{100 * s['clean_gap_at_best']:.2f} pp", "Val. CE minimum at epoch": s["val_ce_min_epoch"],
                     "Val. CE rise after minimum": f"{100 * s['val_ce_rise']:+.0f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def view_init():
    st.title("Initialization")
    lede("He initialization scales weights so that each ReLU layer passes on the signal at the same strength. The Keras default, "
         "Glorot uniform, was derived for symmetric activations and lets it shrink a little at every ReLU layer. "
         "How much that matters depends on depth.")
    st.subheader("Signal strength through depth")
    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        width = st.select_slider("Hidden width", [32, 64, 128, 256, 512, 1024], value=128)
        depth = st.slider("Number of hidden ReLU layers", 1, 30, 2)
        fan0 = st.number_input("Input features", 16, 4096, 784, step=16)
    rel = relative_signal(width, depth, fan0)
    with c2:
        fig = go.Figure()
        layers_x = list(range(depth + 1))
        fig.add_scatter(x=layers_x, y=[1.0] * (depth + 1), mode="lines+markers", name="He normal",
                        line=dict(color=color("he_init"), width=2.6))
        fig.add_scatter(x=layers_x, y=rel, mode="lines+markers", name="Glorot uniform",
                        line=dict(color=color("baseline"), width=2.6))
        fig.add_vline(x=2, line_dash="dot", line_color=THREAD, annotation_text="this network", annotation_font_color=THREAD)
        fig.update_yaxes(type="log", title="Activation RMS relative to He")
        fig.update_xaxes(title="Hidden layer", dtick=1 if depth <= 15 else 5)
        show(style(fig, 380))
    keep = rel[-1]
    note(f"With {depth} hidden layer{'s' if depth > 1 else ''} of width {width}, Glorot keeps about {100 * keep:.1f}% of the "
         f"activation scale that He keeps by the last hidden layer. Each extra hidden-to-hidden layer halves the second moment, "
         "so the gap is mild at depth 2 and severe at depth 20 (Glorot & Bengio, 2010; He et al., 2015).")

    I = A["init"]
    if I:
        seam()
        st.subheader("What was measured in the two untrained networks")
        a, b = st.columns(2, gap="large")
        with a:
            fig = go.Figure()
            for k, nm in [("baseline", "Glorot"), ("he_init", "He")]:
                fig.add_scatter(x=["Input", "Hidden 1", "Hidden 2"], y=I[k]["rms_measured"], mode="lines+markers", name=f"{nm} measured",
                                line=dict(color=color(k), width=2.6))
                fig.add_scatter(x=["Input", "Hidden 1", "Hidden 2"], y=I[k]["rms_predicted"], mode="markers", name=f"{nm} predicted",
                                marker=dict(symbol="diamond-open", size=13, color=color(k), line=dict(width=2)))
            fig.update_yaxes(title="Activation RMS")
            show(style(fig, 380, title="Theory and measurement agree"))
        with b:
            layer = st.radio("Weights of", ["W1", "W2"], horizontal=True, format_func=lambda s: "hidden layer 1" if s == "W1" else "hidden layer 2")
            fig = go.Figure()
            for k, nm in [("baseline", "Glorot uniform"), ("he_init", "He normal")]:
                hst = I[k]["hist"][layer]
                edges = np.array(hst["edges"]); centers = (edges[:-1] + edges[1:]) / 2
                fig.add_bar(x=centers, y=hst["counts"], name=nm, marker=dict(color=color(k)), opacity=0.6, width=float(edges[1] - edges[0]))
            fig.update_layout(barmode="overlay"); fig.update_xaxes(title="Initial weight value"); fig.update_yaxes(title="Count")
            show(style(fig, 330, title="Initial weight distributions"))
    seam()
    st.subheader("The first ten epochs")
    a, b = st.columns(2, gap="large")
    ep0 = list(range(0, 11))
    for col, key, ttl, fmt in [(a, "val_acc", "Validation accuracy from the untrained network", ".0%"),
                               (b, "clean_train_ce", "Training cross-entropy (same 10,000 images each epoch)", None)]:
        with col:
            fig = go.Figure()
            for k in ("baseline", "he_init"):
                if k in R:
                    fig.add_scatter(x=ep0, y=R[k]["diag"][key][:11], mode="lines+markers", name=short(k), line=dict(color=color(k), width=2.6))
            if fmt:
                fig.update_yaxes(tickformat=fmt, range=[max(0.0, min(R[k]["diag"][key][1] for k in ("baseline", "he_init")) - 0.04), None])
            else:
                fig.update_yaxes(type="log")
            fig.update_xaxes(title="Epoch")
            show(style(fig, 360, title=ttl))
    note("Initialization changes where optimization starts. After a few epochs Adam's per-parameter step sizes even out the difference, "
         "which is why the two models end up with similar best validation accuracy.")


def view_regularization():
    st.title("Regularization")
    lede("Both regularizers trade some training fit for a smaller gap. The question is how much to trade.")
    t_l2, t_do = st.tabs(["L2 penalty", "Dropout"])
    S = A["stats"]
    with t_l2:
        keys = [k for k in ["he_init", "l2_0.001", "l2_0.01"] if k in R]
        a, b = st.columns(2, gap="large")
        with a:
            fig = go.Figure()
            for k in keys:
                d = R[k]["diag"]
                fig.add_scatter(x=list(range(EPOCHS + 1)), y=np.array(d["sqnorm_h1"]) + np.array(d["sqnorm_h2"]), mode="lines",
                                name=short(k), line=dict(color=color(k), width=2.6))
            fig.update_yaxes(type="log", title="ΣW², hidden layers"); fig.update_xaxes(title="Epoch")
            show(style(fig, 380, title="The penalty holds the weights down"))
        with b:
            choice = st.radio("Penalty strength", [k for k in keys if k != "he_init"], format_func=label, horizontal=True)
            d = R[choice]["diag"]
            fig = go.Figure()
            fig.add_scatter(x=list(range(EPOCHS + 1)), y=d["clean_train_ce"], mode="lines", name="training cross-entropy",
                            stackgroup="one", line=dict(color=color("he_init"), width=0.5))
            fig.add_scatter(x=list(range(EPOCHS + 1)), y=d["l2_penalty"], mode="lines", name="L2 penalty λΣW²",
                            stackgroup="one", line=dict(color=color(choice), width=0.5))
            fig.update_yaxes(title="Contribution to the training objective"); fig.update_xaxes(title="Epoch")
            show(style(fig, 330, title="What the optimizer is actually minimizing"))
            r0 = d["l2_penalty"][0] / d["clean_train_ce"][0]
            note(f"At epoch 0 the penalty is {r0:.2f}× the data loss. When it starts larger than the data loss, "
                 "the optimizer's first job becomes shrinking weights rather than fitting garments.")
        if S is not None:
            maps = [("Pixel variability", np.asarray(S["pixel_std"]))] + [
                (short(k), np.asarray(S[f"w1map__{k}"])) for k in keys if f"w1map__{k}" in S]
            fig = make_subplots(rows=1, cols=len(maps), subplot_titles=[m[0] for m in maps], horizontal_spacing=0.03)
            for i, (_, m) in enumerate(maps, start=1):
                fig.add_trace(go.Heatmap(z=(m / m.max()).reshape(28, 28), colorscale=[[0, BG], [0.45, "#8A3A55"], [1, "#F6C177"]],
                                         showscale=False, hovertemplate="%{z:.2f}<extra></extra>"), row=1, col=i)
                fig.update_yaxes(autorange="reversed", scaleanchor=f"x{i if i > 1 else ''}", showticklabels=False, showgrid=False, row=1, col=i)
                fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=i)
            show(style(fig, 330, legend=False))
            note("Mean absolute first-layer weight per input pixel, each map scaled to its own maximum. "
                 "Under L2 the weights on near-constant border pixels fade out and the map starts to look like the data's own variability map.")
    with t_do:
        dkeys = [k for k in ["dropout_0.2", "dropout_0.4"] if k in R]
        if not dkeys:
            st.info("No dropout runs found.")
            return
        rate = st.radio("Dropout rate", dkeys, format_func=label, horizontal=True)
        h, d = R[rate]["history"], R[rate]["diag"]
        ep = list(range(1, EPOCHS + 1))
        fig = go.Figure()
        fig.add_scatter(x=ep, y=h["accuracy"], mode="lines", name="training, as Keras reports it (dropout on)",
                        line=dict(color=color(rate), width=1.8, dash="dash"))
        fig.add_scatter(x=ep, y=d["clean_train_acc"][1:], mode="lines", name="training, dropout off", line=dict(color=INK, width=2.4))
        fig.add_scatter(x=ep, y=h["val_accuracy"], mode="lines", name="validation", line=dict(color=color(rate), width=3))
        fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="Epoch")
        show(style(fig, 420, title="Three ways to measure training accuracy"))
        below = int(np.sum(np.array(h["accuracy"]) < np.array(h["val_accuracy"])))
        note(f"Keras training accuracy sits below validation accuracy in {below} of {EPOCHS} epochs. During training each prediction "
             "comes from a thinned network and the number is averaged over an epoch of changing weights; validation uses the full "
             "network at the end of the epoch. Measured the same way, training accuracy is above validation accuracy, as it should be.")
        fig = go.Figure()
        for k in ["he_init"] + dkeys:
            g = np.array(R[k]["diag"]["clean_train_acc"][1:]) - np.array(R[k]["history"]["val_accuracy"])
            fig.add_scatter(x=ep, y=g, mode="lines", name=short(k), line=dict(color=color(k), width=2.6))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_yaxes(tickformat=".1%", title="Training (dropout off) − validation"); fig.update_xaxes(title="Epoch")
        show(style(fig, 360, title="Generalization gap"))


def view_early_stopping():
    st.title("Early stopping")
    lede("Every run trained for the full budget, so Keras' EarlyStopping can be replayed for any setting. "
         "Move the patience and watch where each model would have stopped and which weights it would have kept.")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        monitor = st.radio("Monitor", ["val_loss", "val_accuracy"], horizontal=True,
                           format_func=lambda m: "validation loss" if m == "val_loss" else "validation accuracy")
    with c2:
        patience = st.slider("Patience (epochs without improvement)", 1, 15, 5)
    mode = "min" if monitor == "val_loss" else "max"
    fig = go.Figure(); rows = []
    for k in MAIN:
        h = R[k]["history"]
        stop, restored = simulate_early_stopping(h[monitor], patience, mode)
        best = R[k]["summary"]["best_val_acc"]
        fig.add_scatter(x=list(range(1, EPOCHS + 1)), y=h["val_accuracy"], mode="lines", name=short(k), legendgroup=k,
                        line=dict(color=color(k), width=2.2), opacity=0.9)
        fig.add_scatter(x=[restored], y=[h["val_accuracy"][restored - 1]], mode="markers", showlegend=False, legendgroup=k,
                        marker=dict(symbol="star", size=17, color=color(k), line=dict(color=BG, width=1.5)),
                        hovertemplate=f"{short(k)}: restored epoch {restored}<br>validation %{{y:.2%}}<extra></extra>")
        fig.add_scatter(x=[stop], y=[h["val_accuracy"][stop - 1]], mode="markers", showlegend=False, legendgroup=k,
                        marker=dict(symbol="line-ns-open", size=22, color=color(k), line=dict(width=3)),
                        hovertemplate=f"{short(k)}: stops after epoch {stop}<extra></extra>")
        rows.append({"Model": label(k), "Stops after epoch": stop, "Restores epoch": restored,
                     "Validation accuracy kept": fmt_pct(h["val_accuracy"][restored - 1]),
                     "Given up vs. best of full run": f"{100 * (best - h['val_accuracy'][restored - 1]):.2f} pp",
                     "Epochs saved": EPOCHS - stop})
    fig.update_yaxes(tickformat=".0%", range=[np.percentile(np.concatenate([R[k]["history"]["val_accuracy"] for k in MAIN]), 3) - 0.01, None])
    fig.update_xaxes(title="Epoch")
    show(style(fig, 470, title="Validation accuracy, with stop points (|) and restored weights (★)"))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    note("Monitoring validation loss stops the unregularized models early, close to their loss minimum, which keeps better-calibrated "
         "probabilities at a small cost in accuracy. Monitoring validation accuracy needs more patience because the curve is noisy.")


def view_tuning():
    st.title("Tuning")
    T = A["trials"]
    if T is None:
        st.info("No Optuna results found. Set RUN_TUNING = True in the notebook, run section E5, and export again.")
        return
    lede("A Bayesian search (Optuna, TPE sampler with median pruning) over the three techniques the assignment studies: "
         "the hidden-layer initializer, the L2 coefficient, and the dropout rate. The architecture, optimizer, learning rate, "
         "batch size, and epoch budget stayed fixed.")
    T = T.copy()
    T["curve"] = T["user_attrs_curve"].map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else [])
    comp = T[T.state == "COMPLETE"]
    imp = A["imp"] or {}
    if imp.get("best_params"):
        bp = imp["best_params"]
        st.markdown(f'<span class="pill">best initializer {bp["init"]}</span><span class="pill">λ = {bp["l2"]:.2e}</span>'
                    f'<span class="pill">dropout p = {bp["dropout"]:.2f}</span>'
                    f'<span class="pill">{len(T)} trials, {int((T.state == "PRUNED").sum())} pruned</span>', unsafe_allow_html=True)
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        fig = go.Figure()
        fig.add_scatter(x=comp.number, y=comp.value, mode="markers", name="complete",
                        marker=dict(color=color("tuned"), size=10, line=dict(color=BG, width=1)),
                        customdata=np.stack([comp.params_init, comp.params_l2, comp.params_dropout], axis=1),
                        hovertemplate="trial %{x}<br>%{y:.2%}<br>%{customdata[0]}, λ %{customdata[1]:.1e}, p %{customdata[2]:.2f}<extra></extra>")
        pr = T[T.state == "PRUNED"]
        if len(pr):
            fig.add_scatter(x=pr.number, y=[max(c) if c else None for c in pr.curve], mode="markers", name="pruned (best before stopping)",
                            marker=dict(color=MUTED, size=8, symbol="x"))
        fig.add_scatter(x=comp.number, y=comp.value.cummax(), mode="lines", name="running best", line=dict(color=THREAD, shape="hv", width=2))
        fig.add_hline(y=F["best_val_acc"], line_dash="dot", line_color=color(SEL), annotation_text=f"Step 6 model ({short(SEL)})",
                      annotation_font_color=color(SEL))
        fig.update_yaxes(tickformat=".1%", range=[comp.value.quantile(0.1) - 0.01, comp.value.max() + 0.004]); fig.update_xaxes(title="Trial")
        show(style(fig, 400, title="Optimization history"))
    with b:
        if imp.get("importance"):
            items = sorted(imp["importance"].items(), key=lambda kv: kv[1])
            fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker=dict(color=color("tuned"))))
            fig.update_xaxes(title="Share of explained variance")
            show(style(fig, 400, legend=False, title=f"Importance ({imp.get('method', '')})"))
    symbols = {"glorot_uniform": "circle", "he_normal": "square", "he_uniform": "diamond"}
    fig = go.Figure()
    for init, sym in symbols.items():
        sub = comp[comp.params_init == init]
        if len(sub):
            fig.add_trace(go.Scatter3d(x=np.log10(sub.params_l2), y=sub.params_dropout, z=sub.value, mode="markers", name=init,
                                       marker=dict(size=6, symbol=sym, color=sub.value, colorscale="Viridis",
                                                   cmin=comp.value.quantile(0.1), cmax=comp.value.max(), line=dict(color=BG, width=0.5)),
                                       hovertemplate="log10 λ %{x:.2f}<br>p %{y:.2f}<br>validation %{z:.2%}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis=dict(title="log10 λ", gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)"),
                                 yaxis=dict(title="dropout p", gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)"),
                                 zaxis=dict(title="validation accuracy", tickformat=".1%", gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)")))
    show(style(fig, 560, title="The regularization landscape (drag to rotate)"))
    inits = sorted(comp.params_init.unique())
    fig = go.Figure(go.Parcoords(
        line=dict(color=comp.value, colorscale="Viridis", showscale=True),
        dimensions=[dict(label="initializer", values=comp.params_init.map({v: i for i, v in enumerate(inits)}),
                         tickvals=list(range(len(inits))), ticktext=inits),
                    dict(label="log10 λ", values=np.log10(comp.params_l2)),
                    dict(label="dropout p", values=comp.params_dropout),
                    dict(label="best epoch", values=comp.user_attrs_best_epoch),
                    dict(label="validation accuracy", values=comp.value)],
        labelfont=dict(color=INK), tickfont=dict(color=MUTED), rangefont=dict(color=MUTED)))
    show(style(fig, 420, legend=False, title="Every completed trial, axis by axis (drag along an axis to filter)"))
    fig = go.Figure()
    for _, row in T.iterrows():
        if row.curve:
            pruned = row.state == "PRUNED"
            fig.add_scatter(x=list(range(1, len(row.curve) + 1)), y=row.curve, mode="lines", showlegend=False,
                            line=dict(color=MUTED if pruned else color("tuned"), width=1 if pruned else 1.6), opacity=0.55,
                            hovertemplate=f"trial {int(row.number)} ({row.state.lower()})<br>epoch %{{x}}: %{{y:.2%}}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", range=[comp.value.quantile(0.1) - 0.03, comp.value.max() + 0.005]); fig.update_xaxes(title="Epoch")
    show(style(fig, 380, legend=False, title="Validation curve of every trial (grey lines were pruned)"))


def view_final():
    st.title("Final model")
    lede(f"{F['label']}, restored to epoch {F['best_epoch']}. Everything on this page about the test set comes from one evaluation "
         "of that frozen model.")
    cm = np.array(F["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        txt = np.where(cmn >= 0.005, np.round(100 * cmn, 1).astype(str), "")
        fig = go.Figure(go.Heatmap(z=100 * cmn, x=CLASS_SHORT, y=CLASS_SHORT, text=txt, texttemplate="%{text}", textfont=dict(size=11),
                                   colorscale=[[0, PANEL], [0.06, "#33457A"], [0.4, "#8A6A3A"], [1, THREAD]],
                                   colorbar=dict(title="% of row", tickfont=dict(color=MUTED)),
                                   hovertemplate="true %{y}, predicted %{x}<br>%{z:.1f}% of true class<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title="True class"); fig.update_xaxes(title="Predicted class")
        show(style(fig, 520, legend=False, title=f"Confusion matrix, test accuracy {fmt_pct(F['test_acc'])}"))
    with b:
        pc = pd.DataFrame(F["per_class"]).sort_values("F1")
        fig = go.Figure()
        for col, sym, c in [("Precision", "circle", color("he_init")), ("Recall", "square", color("dropout_0.2")), ("F1", "diamond", INK)]:
            fig.add_scatter(x=pc[col], y=pc["class"], mode="markers", name=col, marker=dict(symbol=sym, size=12, color=c))
        fig.update_xaxes(tickformat=".0%")
        show(style(fig, 520, title="Per-class precision, recall, F1"))
    seam()
    st.subheader("Look inside a cell of the confusion matrix")
    c1, c2 = st.columns(2)
    with c1:
        t = st.selectbox("True class", list(range(10)), index=6, format_func=lambda i: NAMES[i])
    with c2:
        default_p = int(np.argsort(-cm[t] + np.eye(10)[t] * 1e9)[0])
        p_ = st.selectbox("Predicted as", list(range(10)), index=default_p, format_func=lambda i: NAMES[i])
    if A["preds"] is not None:
        probs, yt, Xt = A["preds"]["test_probs_final"], A["preds"]["y_test"], A["X_test"]
        idx = np.where((yt == t) & (probs.argmax(1) == p_))[0]
        idx = idx[np.argsort(-probs[idx].max(1))][:24]
        if len(idx):
            st.image(mosaic([Xt[i] for i in idx], cols=12, scale=3, gap=5))
            note(f"{int(cm[t, p_])} test images labeled {NAMES[t]} were predicted as {NAMES[p_]}; up to 24 are shown, most confident first.")
        elif A["slim"]:
            note("This deployment keeps only misclassified test images to save space, so correct predictions are not shown.")
        else:
            note("No test images fall in this cell.")
    seam()
    a, b = st.columns(2, gap="large")
    with a:
        rel = F["test_reliability"]
        edges = np.array(rel["edges"]); centers = (edges[:-1] + edges[1:]) / 2; cnt = np.array(rel["count"])
        m = cnt > 0
        fig = go.Figure()
        fig.add_bar(x=centers[m], y=np.array(rel["acc"])[m], width=(edges[1] - edges[0]) * 0.9, name="accuracy in bin",
                    marker=dict(color=color(SEL)), customdata=cnt[m], hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2%}<br>%{customdata} images<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration", line=dict(color=THREAD, dash="dash"))
        fig.update_xaxes(range=[0, 1], title="Confidence"); fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(style(fig, 400, title=f"Calibration on the test set, ECE {100 * rel['ece']:.2f}%"))
    with b:
        C = A["cal"]
        if C is not None:
            fig = go.Figure()
            for which, op, nm in [("best", 1.0, "best epoch"), ("final", 0.45, f"epoch {EPOCHS}")]:
                sub = C[C.weights == which].set_index("model").reindex(MAIN)
                fig.add_bar(y=[short(k) for k in MAIN], x=100 * sub.ece, orientation="h", name=nm, opacity=op,
                            marker=dict(color=[color(k) for k in MAIN]))
            fig.update_layout(barmode="group"); fig.update_xaxes(title="Expected calibration error, validation (%)")
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 400, title="Overfitting shows up in confidence first"))
    if A["tsne"] is not None:
        seam()
        st.subheader("What the second hidden layer encodes")
        E = A["tsne"]
        mk = st.radio("Model", list(E.model.unique()), format_func=label, horizontal=True)
        sub = E[E.model == mk]
        fig = go.Figure()
        for c in range(10):
            s = sub[sub.label == c]
            fig.add_scatter(x=s.x, y=s.y, mode="markers", name=NAMES[c], marker=dict(size=5, color=CLASS_COLORS[c], opacity=0.8),
                            hovertemplate=f"{NAMES[c]}<extra></extra>")
        fig.update_xaxes(showticklabels=False, showgrid=False); fig.update_yaxes(showticklabels=False, showgrid=False)
        show(style(fig, 560, title="t-SNE of 2,000 validation images"))
        if A["sil"] is not None:
            note("Silhouette of the hidden representation: " + ", ".join(
                f"{short(r.model)} {r.silhouette:.3f}" for r in A["sil"].itertuples() if r.model in R) +
                 ". t-SNE preserves neighborhoods, not distances between clusters.")
    if A["seeds"] is not None:
        seam()
        st.subheader("Does the ranking survive another seed?")
        Sd = A["seeds"]
        fig = go.Figure()
        for k in [k for k in ALL if k in set(Sd.model)]:
            v = Sd[Sd.model == k]
            fig.add_scatter(x=[short(k)] * len(v), y=v.best_val_acc, mode="markers", name=short(k), showlegend=False,
                            marker=dict(size=13, color=color(k), line=dict(color=BG, width=1.5)), customdata=v.seed,
                            hovertemplate="seed %{customdata}<br>%{y:.2%}<extra></extra>")
            fig.add_scatter(x=[short(k)], y=[v.best_val_acc.mean()], mode="markers", showlegend=False,
                            marker=dict(symbol="line-ew-open", size=34, color=INK, line=dict(width=2)), hoverinfo="skip")
        fig.update_yaxes(tickformat=".1%", title="Best validation accuracy")
        show(style(fig, 400, legend=False))
        note("Dots are seeds (same data split, different initialization, shuffling, and dropout masks); the bar is the mean.")
    if A["mc"] is not None:
        st.subheader("Paired McNemar tests on validation data")
        st.dataframe(A["mc"], hide_index=True)
        note("The selected model was chosen on this same validation set, so these p-values are descriptive.")


# ---- Try it on -------------------------------------------------------------------------------
def prepare_upload(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:                       # Fashion-MNIST garments are light on a dark background
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img.thumbnail((28, 28), resample)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return np.asarray(canvas, dtype=np.float32).ravel() / 255.0


@st.cache_data(show_spinner="Running the sweep")
def robustness_sweep(path_str, kind, n=1000):
    AA = load_artifacts(path_str)
    X, y = AA["X_val"][:n], AA["preds"]["y_val"][:n]
    levels = {"Gaussian noise (σ)": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "Occluded square (pixels)": [0, 4, 8, 12, 16, 20],
              "Horizontal shift (pixels)": [0, 1, 2, 3, 4, 5, 6], "Contrast (× brightness)": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]}[kind]
    rows = []
    for lv in levels:
        if kind.startswith("Gaussian"):
            Xp = np.clip(X + lv * np.random.default_rng(0).standard_normal(X.shape).astype(np.float32), 0, 1)
        else:
            kw = {"Occluded square (pixels)": {"occ": int(lv)}, "Horizontal shift (pixels)": {"dx": int(lv)},
                  "Contrast (× brightness)": {"contrast": float(lv)}}[kind]
            Xp = np.stack([perturb(x, **kw) for x in X])
        for k, w in AA["weights"].items():
            rows.append({"model": k, "level": lv, "accuracy": float(np.mean(forward(w, Xp).argmax(1) == y))})
    return pd.DataFrame(rows)


def view_try():
    st.title("Try it on")
    lede("Take a garment, alter it, and see how every model responds. All predictions are computed live from the exported weights.")
    if not A["weights"] or A["preds"] is None:
        st.info("Weights or images are missing from the artifacts folder.")
        return
    Xv, yv = A["X_val"], A["preds"]["y_val"]
    src = st.radio("Garment", ["From the validation set", "Upload a photo"], horizontal=True)
    true_label = None
    if src == "From the validation set":
        if "idx" not in st.session_state:
            st.session_state.idx = 0
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Pick a random garment"):
                st.session_state.idx = int(np.random.default_rng().integers(0, len(Xv)))
        with c2:
            st.session_state.idx = int(st.number_input("Validation image index", 0, len(Xv) - 1, st.session_state.idx))
        base = Xv[st.session_state.idx]
        true_label = int(yv[st.session_state.idx])
        seed = st.session_state.idx
    else:
        up = st.file_uploader("Photo of one garment on a plain background", type=["png", "jpg", "jpeg", "webp"])
        if up is None:
            note("Upload a photo to start. The image is converted to grayscale, inverted if the background is light, "
                 "and fitted into 28 × 28 pixels like the training data. Real photos look quite different from Fashion-MNIST, "
                 "so expect less confident answers.")
            return
        base, seed = prepare_upload(up), 0

    with st.expander("Alterations", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            noise = st.slider("Gaussian noise σ", 0.0, 0.6, 0.0, 0.02)
            contrast = st.slider("Brightness", 0.2, 1.5, 1.0, 0.05)
        with c2:
            occ = st.slider("Occluded square size", 0, 20, 0)
            occ_x = st.slider("Square position, across", 0, 27, 14)
            occ_y = st.slider("Square position, down", 0, 27, 14)
        with c3:
            dx = st.slider("Shift right", -6, 6, 0)
            dy = st.slider("Shift down", -6, 6, 0)
    x = perturb(base, noise, occ, occ_x, occ_y, dx, dy, contrast, seed)

    a, b = st.columns([1, 2.6], gap="large")
    with a:
        st.image(mosaic([base, x], cols=2, scale=6, gap=10))
        note("Original and altered." + (f" Labeled {NAMES[true_label]}." if true_label is not None else ""))
    keys = [k for k in ALL if k in A["weights"]]
    P = np.stack([forward(A["weights"][k], x[None, :])[0] for k in keys])
    with b:
        fig = go.Figure(go.Heatmap(z=P, x=CLASS_SHORT, y=[short(k) for k in keys], zmin=0, zmax=1, text=np.round(100 * P).astype(int),
                                   texttemplate="%{text}", colorscale=[[0, PANEL], [0.3, "#33457A"], [1, THREAD]], showscale=False,
                                   hovertemplate="%{y}: %{x} %{z:.1%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 330, legend=False, title="Probability each model gives each class (%)"))
        verdicts = []
        for k, p in zip(keys, P):
            pred = int(p.argmax())
            mark = "" if true_label is None else (" ✓" if pred == true_label else " ✗")
            verdicts.append(f'<span class="pill" style="border-color:{color(k)}">{short(k)}: {NAMES[pred]} {100 * p.max():.0f}%{mark}</span>')
        st.markdown(" ".join(verdicts), unsafe_allow_html=True)
    seam()
    st.subheader("Robustness sweep on 1,000 validation garments")
    kind = st.selectbox("Alteration", ["Gaussian noise (σ)", "Occluded square (pixels)", "Horizontal shift (pixels)", "Contrast (× brightness)"])
    df = robustness_sweep(art_path, kind)
    fig = go.Figure()
    for k in keys:
        s = df[df.model == k]
        fig.add_scatter(x=s.level, y=s.accuracy, mode="lines+markers", name=short(k), line=dict(color=color(k), width=2.6))
    fig.update_yaxes(tickformat=".0%", title="Accuracy"); fig.update_xaxes(title=kind)
    if kind.startswith("Contrast"):
        fig.update_xaxes(autorange="reversed")
    show(style(fig, 430))
    note("None of the models saw altered images during training, so this measures how gracefully each one degrades. "
         "A dense network has no built-in notion of position, which is why even small shifts hurt.")


def view_head_to_head():
    st.title("Head to head")
    lede("Pick two models and compare them on the same validation images. Only the images where they disagree say anything "
         "about which one is better, so that is where this view looks (Dietterich, 1998).")
    P = A["preds"]
    keys = [k for k in ALL if P is not None and f"val_probs__{k}" in P]
    if len(keys) < 2:
        st.info("Validation probabilities for at least two models are needed. Re-export the artifacts from the notebook.")
        return
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Model A", keys, index=keys.index(SEL) if SEL in keys else 0, format_func=label)
    others = [k for k in keys if k != a]
    with c2:
        b = st.selectbox("Model B", others, index=others.index("he_init") if "he_init" in others else 0, format_func=label)
    yv = P["y_val"]
    pa, pb = P[f"val_probs__{a}"].astype(np.float32), P[f"val_probs__{b}"].astype(np.float32)
    ca, cb = pa.argmax(1) == yv, pb.argmax(1) == yv
    nb, nc, pval = mcnemar_exact(ca, cb)
    both, neither = int(np.sum(ca & cb)), int(np.sum(~ca & ~cb))
    st.markdown(f'<span class="pill">{short(a)} {fmt_pct(ca.mean())}</span><span class="pill">{short(b)} {fmt_pct(cb.mean())}</span>'
                f'<span class="pill">{nb + nc} disagreements of {len(yv):,}</span><span class="pill">exact McNemar p = {pval:.3g}</span>',
                unsafe_allow_html=True)
    if pval < 0.05:
        winner = short(a) if nb > nc else short(b)
        note(f"{winner} wins clearly more of the disputed images ({max(nb, nc)} vs. {min(nb, nc)}), more than chance would explain at the 5% level.")
    else:
        note(f"The split of disputed images ({nb} vs. {nc}) is within what chance would produce, so these two models are not "
             "distinguishable on this validation set.")
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        m = np.array([[both, nb], [nc, neither]])
        fig = go.Figure(go.Heatmap(z=m, x=[f"{short(b)} right", f"{short(b)} wrong"], y=[f"{short(a)} right", f"{short(a)} wrong"],
                                   text=m, texttemplate="%{text:,}", textfont=dict(size=16), showscale=False,
                                   colorscale=[[0, PANEL], [0.02, "#33457A"], [1, THREAD]], hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 330, legend=False, title="Agreement table"))
    with right:
        diffs = [float(ca[yv == c].mean() - cb[yv == c].mean()) if (yv == c).any() else 0.0 for c in range(10)]
        fig = go.Figure(go.Bar(x=CLASS_SHORT, y=diffs, marker=dict(color=[color(a) if d >= 0 else color(b) for d in diffs]),
                               hovertemplate="%{x}: %{y:+.2%}<extra></extra>"))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_yaxes(tickformat="+.1%", title=f"{short(a)} − {short(b)} accuracy")
        show(style(fig, 330, legend=False, title="Where each model has the edge, by class"))
    Xv = A["X_val"]
    for mask, cap in [(ca & ~cb, f"{short(a)} right, {short(b)} wrong"), (~ca & cb, f"{short(b)} right, {short(a)} wrong")]:
        idx = np.where(mask)[0][:24]
        st.markdown(f"**{cap}** ({int(mask.sum())} images)")
        if len(idx):
            st.image(mosaic([Xv[i] for i in idx], cols=12, scale=3, gap=5))


def view_hypotheses():
    st.title("Hypotheses")
    results = evaluate_hypotheses(A)
    lede(f"{len(results)} predictions were written down, each with a fixed decision rule, before the full experiment ran. "
         "The verdicts below are computed from the results rather than judged by eye (Nosek et al., 2018).")
    icon = {SUPPORTED: "✅", INCONCLUSIVE: "⚪"}
    counts = {v: sum(r.verdict == v for r in results) for v in ("supported", "not supported", "inconclusive")}
    st.markdown(" ".join(f'<span class="pill">{n} {v}</span>' for v, n in counts.items()), unsafe_allow_html=True)
    for r in results:
        h = r.hypothesis
        with st.expander(f"{icon.get(r.verdict, '❌')}  {h.id}. {h.topic}: {r.verdict}"):
            st.markdown(f"**Prediction.** {h.statement}\n\n**Decision rule.** {h.rule}\n\n**Evidence.** {r.evidence}.\n\n"
                        f"**Conclusion.** {r.conclusion}\n\n*Basis: {h.basis}.*")
    st.download_button("Download the full results report (Markdown)", build_report(A, fig_rel="figures"), "RESULTS.md", "text/markdown")


def view_registry():
    st.title("Run registry")
    lede("Every training run in the cache, with a fingerprint of its configuration. Runs that share a fingerprint differ only in their seed.")
    Rg = A["registry"]
    if Rg is None or not len(Rg):
        st.info("No registry found. Run `python -m fitting_room registry` after training, or keep artifacts/runs next to the app.")
        return
    models = sorted(Rg.model.unique())
    pick = st.multiselect("Models", models, default=models, format_func=lambda k: label(k) if k in R else k)
    sub = Rg[Rg.model.isin(pick)]
    fig = go.Figure()
    for k in pick:
        s = sub[sub.model == k]
        fig.add_scatter(x=s.clean_gap_at_best, y=s.best_val_acc, mode="markers", name=short(k) if k in R else k,
                        marker=dict(size=13, color=color(k) if k in R else THREAD, line=dict(color=BG, width=1.5)),
                        customdata=s.seed, hovertemplate="seed %{customdata}<br>gap %{x:.2%}<br>validation %{y:.2%}<extra></extra>")
    fig.update_xaxes(tickformat=".1%", title="Clean train − validation gap at the best epoch")
    fig.update_yaxes(tickformat=".1%", title="Best validation accuracy")
    show(style(fig, 440))
    note("Up and to the left is the goal: high validation accuracy with a small gap.")
    g = sub.groupby(["model", "config_hash"]).best_val_acc.agg(["count", "mean", "std"]).reset_index()
    g["mean"] = g["mean"].map(fmt_pct)
    g["std"] = g["std"].map(lambda v: "" if pd.isna(v) else f"{100 * v:.2f} pp")
    st.dataframe(g.rename(columns={"count": "runs", "mean": "mean best val. accuracy", "std": "SD"}), hide_index=True)
    st.dataframe(sub, hide_index=True)
    st.download_button("Download registry (CSV)", sub.to_csv(index=False), "registry.csv", "text/csv")


VIEWS = {"Overview": view_overview, "The garments": view_garments, "Learning curves": view_curves, "Initialization": view_init,
         "Regularization": view_regularization, "Early stopping": view_early_stopping, "Tuning": view_tuning,
         "Final model": view_final, "Head to head": view_head_to_head, "Hypotheses": view_hypotheses,
         "Run registry": view_registry, "Try it on": view_try}
VIEWS[page]()
