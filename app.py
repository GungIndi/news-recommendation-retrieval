from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model and data once
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
df = pd.read_pickle("embedded_all.pkl")


# Preprocess input
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Get top matches
def get_top_matches(query, df, model, top_k=5, alpha=0.5):
    cleaned_text = preprocess_query(query)
    new_embedding = model.encode([cleaned_text])

    embed_full = np.vstack(df["embed_all"].values)
    embed_kw = np.vstack(df["embedding"].values)

    sim_full = cosine_similarity(new_embedding, embed_full)[0]
    sim_kw = cosine_similarity(new_embedding, embed_kw)[0]

    combined_sim = alpha * sim_kw + (1 - alpha) * sim_full
    df["similarity"] = combined_sim

    return df.sort_values(by="similarity", ascending=False).head(top_k)[
        ["judul", "source", "link", "similarity", "content", "waktu"]
    ]


# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    top_k = 5

    if request.method == "POST":
        query = request.form.get("query", "")
        top_k = int(request.form.get("top_k", 5))

        if query.strip():
            results = get_top_matches(query, df.copy(), model, top_k=top_k)
            results["waktu"] = pd.to_datetime(results["waktu"])
            results["waktu"] = results["waktu"].dt.strftime("%d-%m-%Y")

    return render_template("index.html", results=results, query=query, top_k=top_k)


# ...existing code...


@app.route("/similar-news/<int:news_id>")
def similar_news(news_id):
    try:
        # Get the news article
        news_article = df.iloc[news_id]

        # Calculate similarities
        embed_full = np.vstack(df["embed_all"].values)
        news_embed = news_article["embed_all"].reshape(1, -1)
        similarities = cosine_similarity(news_embed, embed_full)[0]

        # Get top 3 similar news (excluding self)
        similar_indices = np.argsort(similarities)[-4:-1][::-1]
        similar_news = df.iloc[similar_indices][
            ["judul", "source", "link", "content", "waktu"]
        ].copy()
        similar_news["similarity"] = similarities[similar_indices]

        # Format the date
        similar_news["waktu"] = pd.to_datetime(similar_news["waktu"])
        similar_news["waktu"] = similar_news["waktu"].dt.strftime("%d-%m-%Y")

        return render_template("_similar_news.html", similar_news=similar_news)
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True)
