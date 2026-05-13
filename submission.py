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
#     "drawdata==0.5.0",
#     "anywidget>=0.9",
#     "seaborn==0.13.2",
#     "altair==6.0.0",
#     "vl-convert-python",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semantic Axes
    ##### modified `assignment.py` for the unis dataset
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2 if you want faster but noisier results
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Reference Implementation

    Two short functions: **build an axis** and **score words**. Copy these
    into your own submission.
    """)
    return


@app.cell
def _(np):
    def make_axis(positive_words, negative_words, embedding_model):
        """Return a unit-length semantic axis from two word sets."""

        # get the embeddings for each pole
        pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
        neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)

        # Compute the pole centroids
        # axis = 0 means "average across the rows, keep the columns (dims) intact"
        # since pos_emb is shape (num_pos_words, embedding_dim), the mean is shape (embedding_dim,)
        pole_pos = pos_emb.mean(axis=0)  # (embedding_dim,)
        pole_neg = neg_emb.mean(axis=0)  # (embedding_dim,)

        # The axis is the difference between the two centroids, normalized to unit length.
        v = pole_pos - pole_neg

        v = v / (np.linalg.norm(v) + 1e-10)  # add small epsilon to prevent division by zero

        return v / (np.linalg.norm(v) + 1e-10)

    return (make_axis,)


@app.function
def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""

    emb = embedding_model.encode(list(words), normalize_embeddings=True)

    # Projection to the axis is just a dot product (since the axis is unit-length).
    # @ is matrix multiplication in NumPy. Since `emb` is shape (num_words, embedding_dim) and `axis` is shape (embedding_dim,), the result is shape (num_words,), which is exactly what we want: one score per word.
    proj = emb @ axis

    return proj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Universities
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/universities.csv")
    print(f"{len(df)} unis of {df['type'].nunique()} types across {df['region'].nunique()} regions.")

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 — Design two semantic axes

    A **good axis** is:

    - **Well-separated**: the + and − word sets should be far apart in
      embedding space (pole distance ≥ 0.3; not a strict requirement but a rule-of-thumb).
    - **Discriminative**: when projected onto your dataset it should spread
      the points out, not pile them in the middle.
    - **Orthogonal to the other axis**: the two axes should capture
      different aspects of the data.

    /// note | Why multiple words per pole — why not just one?

    A single word's embedding is **noisy**. It carries the quirks of how
    that specific word appears in the training data: rare senses,
    collocations, polysemy, branding. If you build an axis from just two
    single words, those quirks *become* the axis.

    Rule of thumb: **3–6 words per pole**. Fewer is too noisy; many more
    starts to dilute the concept by pulling in unrelated vocabulary.

    ///

    Our two axes for cities:

    - **Horizontal** — *small town / village* (−) ↔ *megacity / metropolis* (+)
    - **Vertical** — *cold / northern climate* (−) ↔ *tropical / warm climate* (+)

    These are conceptually independent: Singapore is both tropical *and* a
    megacity; Reykjavik is cold and small; Asunción is warm-ish but not a
    global megalopolis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 1 — small town / rural / suburban (−) ↔ urban / big city / downtown (+)
    """)
    return


@app.cell
def _(make_axis, model):
    axis1_pos = [
        "urban",
        "big city",
        "downtown",
        "center city",
        "metropolitan"
    ]
    axis1_neg = [
        "small town",
        "suburban",
        "rural",
        "isolated",
        "college town"
    ]
    axis_setting = make_axis(axis1_pos, axis1_neg, model)
    return (axis_setting,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 2 — ← vocational/professional(−) ↔ elite/scholarly (+)
    """)
    return


@app.cell
def _(make_axis, model):
    axis2_pos = [
        "research",
        "scholarly",
        "elite",
        "academic"
    ]

    axis2_neg = [
        "technical",
        "professional",
        "open",
        "vocational",
    ]

    axis_style = make_axis(axis2_pos, axis2_neg, model)
    return (axis_style,)


@app.cell
def _(axis_setting, axis_style, df, model):
    x = score_words(df["name"].tolist(), axis_setting, model)
    y = score_words(df["name"].tolist(), axis_style, model)
    df_scored = df.assign(x=x, y=y)
    df_scored.head()
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 — Visualize
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    color_by = mo.ui.dropdown(
        options={
            "region (categorical)": "region",
            "type (categorical)": "type",
        },
        value="type (categorical)",
        label="Color by: ",
    )
    return (color_by,)


@app.cell(hide_code=True)
def _(alt, color_by, df_scored, mo):
    # Okabe–Ito palette — categorical, colorblind-safe.
    REGION_COLORS = {
        "West": "#009E73",
        "South": "#0072B2",
        "Northeast": "#D55E00",
        "Midwest": "#56B4E9",
    }


    TYPE = ['Ivy', 'Elite Private', 'Tech', 'Public Flagship', 'Public Regional', 'Liberal Arts', 'HBCU', 'Religious', 'Womens', 'Service Academy', 'For-Profit', 'Community College', 'Tribal', 'Specialized Arts']

    if color_by.value == "region":
        # Categorical → qualitative palette.
        _color = alt.Color(
            "region:N",
            scale=alt.Scale(
                domain=list(REGION_COLORS.keys()),
                range=list(REGION_COLORS.values()),
            ),
            legend=alt.Legend(title="Region"),
        )
    elif color_by.value == "type":
        # Ordinal → sequential palette, sorted along the order.
        _color = alt.Color(
            "type:O",
            sort=TYPE,
            scale=alt.Scale(domain=TYPE, scheme="viridis"),
            legend=alt.Legend(title="Type"),
        )

    chart = (
        alt.Chart(df_scored)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "x:Q",
                title="←  small town | urban →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "y:Q",
                title="← vocational/professional | elite/scholarly →",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            color=_color,
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("x:Q", title="setting score", format=".3f"),
                alt.Tooltip("y:Q", title="focus/style score", format=".3f"),
                alt.Tooltip("name:N", title="University"),
               ],
        )
        .properties(
            width=720,
            height=500,
            title="US Universities in a 2D semantic space (hover for details)",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()  # pan + zoom
    )

    chart.save('figs/uni_semaxis.png')

    # Stack the dropdown directly above the chart so it is always visible.
    mo.vstack([color_by, chart])
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
    import altair as alt
    from sentence_transformers import SentenceTransformer
    from drawdata import ScatterWidget

    return SentenceTransformer, alt, mo, np, pd


@app.function(hide_code=True)
def make_preset_clusters(n: int = 25, seed: int = 0):
    """Four 2-D Gaussian blobs used when the widget is empty.

    Colors match drawdata's first four pens:
      1. blue   (#1f77b4) — − pole
      2. red    (#d62728) — + pole
      3. green  (#2ca02c) — test points
      4. orange (#ff7f0e) — test points
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    clusters = [
        ("#1f77b4", [140, 260]),
        ("#d62728", [360, 140]),
        ("#2ca02c", [240, 320]),
        ("#ff7f0e", [300, 210]),
    ]

    xs, ys, cs = [], [], []
    for color, loc in clusters:
        pts = rng.normal(loc=loc, scale=[32, 28], size=(n, 2))
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
        cs.extend([color] * n)

    return pd.DataFrame({"x": xs, "y": ys, "color": cs})


@app.function(hide_code=True)
def plot_semaxis_2d(df):
    """Interactive SemAxis demo.

    df is expected to have columns x, y, color. The first two unique color
    values are treated as the negative and positive poles respectively; any
    additional colors are shown as "test" classes that get projected onto
    the same axis.

    Left panel: points + bold arrows from the origin to each pole
    centroid, plus the thick SemAxis arrow (e_+ - e_-).
    Right panel: per-class 1-D projection scores as a seaborn violin
    with an overlaid strip.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="talk", font_scale=0.85)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    colors = df["color"].unique().tolist() if len(df) else []
    if len(colors) < 2:
        for a in (ax1, ax2):
            a.text(
                0.5,
                0.5,
                "Draw points with at least two colors, or the preset will appear.",
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
                "Pole centroids coincide - move the two pole clusters apart.",
                ha="center",
                va="center",
                transform=a.transAxes,
                color="#666",
            )
            a.set_axis_off()
        return fig
    axis_unit = v / v_len

    # Projections measured from the ORIGIN (matches the real SemAxis algorithm).
    t = pts @ axis_unit

    # ---- Left: 2-D scene ----
    class_labels = [f"class {i + 1}" for i in range(len(colors))]
    df_plot = df.copy()
    df_plot["class"] = df_plot["color"].map(dict(zip(colors, class_labels)))
    palette = dict(zip(class_labels, colors))

    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="class",
        palette=palette,
        s=70,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        ax=ax1,
        legend=False,
        zorder=2,
    )

    # Thin guide lines from origin to each pole centroid.
    for center, c_rgb, lbl in [
        (neg_c, neg_color, r"$e_{-}$ (- pole centroid)"),
        (pos_c, pos_color, r"$e_{+}$ (+ pole centroid)"),
    ]:
        ax1.annotate(
            "",
            xy=center,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-",
                color=c_rgb,
                lw=1,
                alpha=0.6,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
        ax1.plot([], [], color=c_rgb, lw=1, alpha=0.6, label=lbl)

    # SemAxis arrow: e_+ - e_-.
    ax1.annotate(
        "",
        xy=pos_c,
        xytext=neg_c,
        arrowprops=dict(
            arrowstyle="-|>",
            color="#222",
            lw=2.5,
            mutation_scale=20,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )
    ax1.plot([], [], color="#222", lw=2.5, label=r"SemAxis: $e_{+} - e_{-}$")

    # Origin marker.
    ax1.scatter([0], [0], s=55, marker="x", color="#222", linewidths=2, zorder=5)
    ax1.annotate(
        "origin",
        xy=(0, 0),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=9,
        color="#555",
    )

    pad_x = max(30.0, 0.1 * (pts[:, 0].max() - pts[:, 0].min()))
    pad_y = max(30.0, 0.1 * (pts[:, 1].max() - pts[:, 1].min()))
    ax1.set_xlim(min(0.0, pts[:, 0].min()) - pad_x, max(0.0, pts[:, 0].max()) + pad_x)
    ax1.set_ylim(min(0.0, pts[:, 1].min()) - pad_y, max(0.0, pts[:, 1].max()) + pad_y)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Pole centroids are vectors; the SemAxis is their difference", pad=12)
    ax1.legend(loc="best", fontsize=9, frameon=False)
    sns.despine(ax=ax1)

    # ---- Right: 1-D projected scores per class ----
    proj_df = pd.DataFrame(
        {
            "projection": t,
            "class": pd.Categorical(
                [class_labels[colors.index(c)] for c in color_arr],
                categories=class_labels,
                ordered=True,
            ),
        }
    )

    sns.stripplot(
        data=proj_df,
        x="projection",
        y="class",
        hue="class",
        palette=palette,
        size=5,
        jitter=0.2,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=ax2,
        legend=False,
    )

    # Per-class mean as a short vertical tick.
    means = proj_df.groupby("class", observed=True)["projection"].mean()
    for i, lbl in enumerate(class_labels):
        if lbl in means.index:
            ax2.plot(
                [means[lbl], means[lbl]],
                [i - 0.28, i + 0.28],
                color="#222",
                lw=1.5,
                zorder=5,
            )

    ax2.set_xlabel(r"projection onto SemAxis $\rightarrow$")
    ax2.set_ylabel("")
    ax2.set_title("1-D projected scores per class", pad=12)
    sns.despine(ax=ax2, left=True)
    ax2.tick_params(axis="y", length=0)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    app.run()
