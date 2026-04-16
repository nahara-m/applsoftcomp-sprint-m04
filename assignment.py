# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.10.0",
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "scipy>=1.11",
#     "ipython>=8.0",
#     "drawdata>=0.3.6",
#     "anywidget>=0.9",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semantic Axes: Intro Notebook

    /// Note | This notebook is NOT the deliverable.

    It is a worked example that walks through the full pipeline you are expected to build in your own submission.

    Feel free to copy the `make_axis`, `score_words`, and `center_scores`
    functions from this notebook into your submission — they are the
    reference implementation.

    ///

    We'll learn **SemAxis** (An, Kwak, and Ahn. 2019 ACL), a tool to create interpretable window into embedding space.

    > Jisun An, Haewoon Kwak, and Yong-Yeol Ahn. 2018. SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2450–2461, Melbourne, Australia. Association for Computational Linguistics.


    The original work uses word embeddings. Here we use **Sentence Transformers** to create word embeddings. Let us first load the sentence transformer model. We'll then build the concept of SemAxis and how we implement it using sentence transformers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Setup — Load the embedding model

    The sentence transformer is a small (~90 MB) pre-trained model that
    maps any text to a 384-dimensional unit vector. The first call downloads
    the weights; subsequent calls reuse them from disk.

    **Copy this cell into your own submission** — you will need the same
    model (or any other sentence transformer) to reproduce anything below.
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — Concepts

    Run the cells below in order. Each takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 1 — Word Embeddings

    A sentence transformer maps any text to a point in a high-dimensional
    space. Texts with similar meaning land near each other.
    """)
    return


@app.cell
def _(model):
    _words = ["Harvard University", "MIT", "Swarthmore College", "community college"]
    _emb = model.encode(_words, normalize_embeddings=True)
    print(f"Each text → a vector of {_emb.shape[1]} numbers.\n")
    for _w, _e in zip(_words, _emb):
        print(f"  '{_w}' (first 6 dims): {_e[:6].round(3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 2 — Cosine Similarity

    Two unit vectors are **similar** when they point in the same direction.
    We measure this with a dot product (= cosine similarity for unit vectors).
    """)
    return


@app.cell
def _(model):
    _pairs = [
        ("Harvard University", "Yale University"),
        ("MIT", "Georgia Tech"),
        ("Harvard University", "community college"),
    ]
    _e = model.encode([w for p in _pairs for w in p], normalize_embeddings=True)
    for _i, (_a, _b) in enumerate(_pairs):
        print(f"  {_a:<30} ↔  {_b:<25}  sim = {_e[2 * _i] @ _e[2 * _i + 1]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concept 3 — SemAxis

    A **semantic axis** is a direction in embedding space defined by two
    opposite word sets. Any new word can be scored by projecting its
    embedding onto that direction.

    $$
    \text{axis} = \frac{\bar{\mathbf{e}}_{+} - \bar{\mathbf{e}}_{-}}
                       {\|\bar{\mathbf{e}}_{+} - \bar{\mathbf{e}}_{-}\|}
    \qquad
    \text{score}(w) = \mathbf{e}_w \cdot \text{axis}
    $$

    Large positive → near the + pole. Large negative → near the − pole.
    Near zero → (roughly) orthogonal to the axis, i.e. the axis is not
    informative for this word.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Try it in 2D

    The math above lives in 384-dim space, which is impossible to draw.
    The same operation in **2D** is easy to see. Use the widget below:

    - **Draw two clusters** with the first two pens. The first color is
      treated as the **− pole**, the second as the **+ pole**. If you
      leave it blank, a preset appears below.
    - The solid arrow is the **SemAxis**: the unit vector from the − pole
      centroid to the + pole centroid.
    - Dashed segments show each point's **projection** onto the axis.
    - The right panel plots those projections as 1D scores.

    Now imagine that the two colored clusters are your positive and negative
    *pole words* — the picture on the right is exactly what `score_words`
    returns in high-dimensional space.
    """)
    return


@app.cell
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget(width=520, height=380))
    widget
    return (widget,)


@app.function
def make_preset_clusters(n: int = 25, seed: int = 0):
    """Two 2-D Gaussian blobs used when the widget is empty."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    neg = rng.normal(loc=[140, 260], scale=[35, 28], size=(n, 2))
    pos = rng.normal(loc=[360, 140], scale=[35, 28], size=(n, 2))
    return pd.DataFrame(
        {
            "x": list(neg[:, 0]) + list(pos[:, 0]),
            "y": list(neg[:, 1]) + list(pos[:, 1]),
            "color": ["#1f77b4"] * n + ["#d62728"] * n,
        }
    )


@app.function
def plot_semaxis_2d(df):
    """2-D scatter with the semaxis line + projections, and a 1-D strip plot.

    df is expected to have columns x, y, color. The first two unique color
    values are treated as the negative and positive poles respectively.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    colors = df["color"].unique().tolist() if len(df) else []
    if len(colors) < 2:
        for a in (ax1, ax2):
            a.text(
                0.5,
                0.5,
                "Draw points with two colors, or the preset will appear.",
                ha="center",
                va="center",
                transform=a.transAxes,
                color="#666",
            )
            a.set_axis_off()
        return fig

    neg_color, pos_color = colors[0], colors[1]
    neg = df.loc[df["color"] == neg_color, ["x", "y"]].to_numpy()
    pos = df.loc[df["color"] == pos_color, ["x", "y"]].to_numpy()
    pts = df[["x", "y"]].to_numpy()
    color_arr = df["color"].to_numpy()

    neg_c = neg.mean(axis=0)
    pos_c = pos.mean(axis=0)
    v = pos_c - neg_c
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-8:
        for a in (ax1, ax2):
            a.text(
                0.5,
                0.5,
                "Cluster centers coincide — move the clusters apart.",
                ha="center",
                va="center",
                transform=a.transAxes,
                color="#666",
            )
            a.set_axis_off()
        return fig
    axis_unit = v / v_len
    midpoint = (pos_c + neg_c) / 2

    # Scalar projection onto axis (relative to midpoint).
    t = (pts - midpoint) @ axis_unit
    proj_xy = midpoint[None, :] + t[:, None] * axis_unit[None, :]

    # ---- Left: 2D scene ----
    # Axis as an arrow pointing from − to +, extended past both centers.
    span = np.array([-1.4, 1.4]) * v_len
    ep = midpoint[None, :] + span[:, None] * axis_unit[None, :]
    ax1.annotate(
        "",
        xy=ep[1],
        xytext=ep[0],
        arrowprops=dict(arrowstyle="->", color="#333", lw=2),
        zorder=1,
    )

    # Dashed projection lines.
    for i in range(len(pts)):
        ax1.plot(
            [pts[i, 0], proj_xy[i, 0]],
            [pts[i, 1], proj_xy[i, 1]],
            color="#bbb",
            linestyle="--",
            linewidth=0.7,
            zorder=1,
        )

    # Points.
    for c in (neg_color, pos_color):
        sub = df[df["color"] == c]
        ax1.scatter(
            sub["x"],
            sub["y"],
            c=c,
            s=55,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    # Cluster centroids highlighted as hollow rings.
    for c_rgb, center, lbl in [
        (neg_color, neg_c, "− pole centroid"),
        (pos_color, pos_c, "+ pole centroid"),
    ]:
        ax1.scatter(
            *center,
            s=380,
            marker="o",
            facecolor="none",
            edgecolor=c_rgb,
            linewidth=2.5,
            zorder=4,
            label=lbl,
        )

    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("2D points, SemAxis direction, and projections")
    ax1.legend(loc="best", fontsize=8, frameon=False)

    # ---- Right: 1D projected scores ----
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.35, 0.35, size=len(t))
    for c in (neg_color, pos_color):
        mask = color_arr == c
        ax2.scatter(
            t[mask],
            jitter[mask],
            c=c,
            s=55,
            edgecolor="white",
            linewidth=0.6,
        )
    ax2.axvline(0, color="#444", linewidth=1)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks([])
    ax2.set_xlabel("projection onto SemAxis  →")
    ax2.set_title("1D projected scores")
    for side in ("left", "top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    return fig


@app.cell
def _(pd, widget):
    # Read the drawn data reactively. Fall back to the preset when empty.
    _ = widget.value  # noqa: register widget as reactivity dependency
    try:
        drawn = widget.data_as_pandas
    except Exception:
        drawn = pd.DataFrame()
    df_demo = drawn if (not drawn.empty and drawn["color"].nunique() >= 2) else make_preset_clusters()
    plot_semaxis_2d(df_demo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Reference Implementation

    Three short functions: **build an axis**, **score words**, and
    **center scores** so that `0` means equidistant from the two poles.
    Copy these into your own submission.
    """)
    return


@app.function
def make_axis(positive_words, negative_words, embedding_model):
    """Return a unit-length semantic axis from two word sets."""
    import numpy as np

    pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)
    v = pos_emb.mean(axis=0) - neg_emb.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-10)


@app.function
def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""
    emb = embedding_model.encode(list(words), normalize_embeddings=True)
    return emb @ axis


@app.function
def center_scores(scores, pos_words, neg_words, axis, embedding_model):
    """Shift raw scores so that 0 = equidistant from the two pole centroids.

    Why: raw scores are projections onto the axis. Score = 0 only means the
    embedding is orthogonal to the axis, not that the word is "neutral".
    Centering places 0 at the midpoint between the two pole centroids,
    which is the interpretation most people expect.
    """
    pos_emb = embedding_model.encode(pos_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(neg_words, normalize_embeddings=True)
    midpoint = (pos_emb.mean(axis=0) @ axis + neg_emb.mean(axis=0) @ axis) / 2
    return scores - midpoint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Worked Example: U.S. Universities

    We load the `data/universities.csv` case study and walk through the full
    pipeline end-to-end. Study this example, then build your own in a
    separate notebook using whichever dataset you choose.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/universities.csv")
    print(f"{len(df)} universities across {df['type'].nunique()} types and {df['region'].nunique()} regions.")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Design two semantic axes

    A **good axis** is:

    - **Well-separated**: the + and − word sets should be far apart in
      embedding space (pole distance ≥ 0.3).
    - **Discriminative**: when projected onto your dataset it should spread
      the points out, not pile them in the middle.

    Tips:

    - Use several words per pole (3–6). Single words are noisy.
    - Institution *names* often work better than abstract concepts, because
      the model has rich context for named entities.
    - If your first axis fails, iterate on the pole words.
    """)
    return


@app.cell
def _(df, model):
    # Axis 1 — research intensity vs teaching focus
    axis1_pos = ["research university", "PhD program", "laboratory", "publications", "grant funding", "doctoral"]
    axis1_neg = ["teaching college", "undergraduate focus", "mentoring", "small classes", "community access"]

    # Axis 2 — religious affiliation vs secular
    axis2_pos = ["Catholic", "Christian", "religious affiliation", "faith-based", "seminary"]
    axis2_neg = ["secular", "public research", "state university", "non-religious"]

    ax1 = make_axis(axis1_pos, axis1_neg, model)
    ax2 = make_axis(axis2_pos, axis2_neg, model)

    raw_x = score_words(df["name"].tolist(), ax1, model)
    raw_y = score_words(df["name"].tolist(), ax2, model)
    x = center_scores(raw_x, axis1_pos, axis1_neg, ax1, model)
    y = center_scores(raw_y, axis2_pos, axis2_neg, ax2, model)

    df_scored = df.assign(x=x, y=y)
    df_scored.head()
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Visualize

    The scatterplot below follows a few data-viz principles you should
    apply in your own submission:

    - **Colorblind-safe palette** (Okabe–Ito). Do not use raw matplotlib
      defaults for categorical color.
    - **Distinct marker shapes** redundantly encode region, so the plot
      is still readable in grayscale or for viewers who confuse colors.
    - **Axis labels are the pole words**, not `x` / `y`. The reader should
      be able to interpret the plot without reading extra text.
    - **Zero lines** draw the eye to the midpoint — Gestalt "common fate"
      groups points on the same side of each axis.
    - **Annotations** on a few extreme points guide pre-attentive attention
      to the story you want the reader to see first.
    """)
    return


@app.cell
def _(df_scored, plt):
    OKABE_ITO = {
        "Ivy": "#E69F00",
        "Elite Private": "#D55E00",
        "Tech": "#0072B2",
        "Public Flagship": "#009E73",
        "Public Regional": "#56B4E9",
        "Liberal Arts": "#CC79A7",
        "HBCU": "#F0E442",
        "Religious": "#999999",
        "Womens": "#AA4499",
        "Service Academy": "#332288",
        "For-Profit": "#882255",
        "Community College": "#117733",
        "Tribal": "#44AA99",
        "Specialized Arts": "#6699CC",
    }
    REGION_MARKERS = {
        "Northeast": "o",
        "South": "s",
        "Midwest": "^",
        "West": "D",
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for region, marker in REGION_MARKERS.items():
        for utype, color in OKABE_ITO.items():
            sub = df_scored[(df_scored["region"] == region) & (df_scored["type"] == utype)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["x"], sub["y"], c=color, marker=marker, s=70, edgecolor="white", linewidth=0.6, alpha=0.9)

    ax.axhline(0, color="#444", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#444", linewidth=0.8, zorder=0)
    ax.set_xlabel("← teaching-focused          research-intensive →", fontsize=11)
    ax.set_ylabel("← secular                  religious →", fontsize=11)
    ax.set_title("U.S. universities in a 2D semantic space", fontsize=13, pad=12)

    # Annotate a few extremes to guide the reader
    for _, row in df_scored.nlargest(3, "x").iterrows():
        ax.annotate(row["name"], (row["x"], row["y"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    for _, row in df_scored.nsmallest(3, "x").iterrows():
        ax.annotate(row["name"], (row["x"], row["y"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    for _, row in df_scored.nlargest(2, "y").iterrows():
        ax.annotate(row["name"], (row["x"], row["y"]), fontsize=8, xytext=(4, 2), textcoords="offset points")

    # Two legends: color = type, shape = region
    from matplotlib.lines import Line2D

    type_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=c, markeredgecolor="white", markersize=8, label=t)
        for t, c in OKABE_ITO.items()
    ]
    region_handles = [
        Line2D([], [], marker=m, linestyle="", markerfacecolor="#777", markeredgecolor="white", markersize=8, label=r)
        for r, m in REGION_MARKERS.items()
    ]
    leg1 = ax.legend(
        handles=type_handles,
        title="Type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=region_handles,
        title="Region",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 — Document what you see

    A good observation paragraph answers:

    1. **What clusters form?** Which groups are pulled apart, which overlap?
    2. **Are there surprises?** Points on the "wrong" side of an axis are
       often the most informative — the model is telling you something
       about how the entity is *discussed*, which may differ from what you
       expect.
    3. **What does the axis *not* capture?** Every axis is a linear
       projection. Some distinctions you care about may be orthogonal to
       both of your axes.

    **Example observation for the plot above:**

    > HBCUs and religious institutions separate cleanly along the vertical
    > axis, while the horizontal axis spreads flagship research universities
    > away from community colleges and regional publics. Service academies
    > cluster near the secular–teaching quadrant, which is consistent with
    > their undergraduate-heavy mission. An unexpected finding: several
    > Ivies score slightly on the "religious" side of axis 2, likely a
    > residue of their historical denominational origins still present in
    > online text. The axes do not separate STEM-heavy from humanities-heavy
    > institutions — that distinction would require a third axis.

    ---

    Now open the **`README.md`** and build your own notebook for one of the
    three case studies (or your own dataset). Your submission is evaluated
    on its pipeline, its git history, its documentation, and the clarity
    of its final figure — not on matching this example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ⚙ Back office

    Infrastructure cells. You do not need to read or modify these.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer
    from drawdata import ScatterWidget

    return ScatterWidget, SentenceTransformer, mo, np, pd, plt


if __name__ == "__main__":
    app.run()
