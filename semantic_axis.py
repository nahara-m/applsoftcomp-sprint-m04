from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import plotly.express as px
import os


model = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2 if you want faster but noisier results

uni_df = pd.read_csv("data/universities.csv")

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



def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""

    emb = embedding_model.encode(list(words), normalize_embeddings=True)

    # Projection to the axis is just a dot product (since the axis is unit-length).
    # @ is matrix multiplication in NumPy. Since `emb` is shape (num_words, embedding_dim) and `axis` is shape (embedding_dim,), the result is shape (num_words,), which is exactly what we want: one score per word.
    proj = emb @ axis

    return proj




# semantic axes
axis1_pos = [
    "urban",
    "big city",
    "downtown",
]
axis1_neg = [
    "small town",
    "suburban",
    "rural",
]


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


axis_setting = make_axis(axis1_pos, axis1_neg, model)
axis_style = make_axis(axis2_pos, axis2_neg, model)

x = score_words(uni_df["name"].tolist(), axis_style, model)
y = score_words(uni_df["name"].tolist(), axis_setting, model)
df_scored_1 = uni_df.assign(x=x, y=y)


#data viz
fig_1 = px.scatter(
    df_scored_1,
    x="x", y="y",
    color="region",
    symbol="type",
    color_discrete_sequence=px.colors.qualitative.Safe,
    hover_name="name",
    hover_data={"x":False, "y":False,"type":False},
    labels={"x": "← vocational/professional | elite/scholarly →", "y":"←  small town | urban →"},
)

os.makedirs("./figs", exist_ok=True)

fig_1.write_html("figs/semantic_axis.html")