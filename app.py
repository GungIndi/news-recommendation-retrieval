from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import json
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score

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


def classify_query_length(query):
    n_words = len(query.strip().split())
    if n_words <= 3:
        return "short"
    elif n_words <= 6:
        return "long"
    else:
        return "extra_long"


def apk(actual, k=5):
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(actual[:k]):
        if p > 0:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / min(k, len(actual)) if actual else 0


# Get top matches
def get_top_matches(
    query, df, model, top_k=5, alpha=0.5, similarity_type="combined_embedding"
):
    cleaned_text = preprocess_query(query)
    new_embedding = model.encode([cleaned_text])

    embed_full = np.vstack(df["embed_all"].values)
    embed_kw = np.vstack(df["embedding"].values)

    sim_full = cosine_similarity(new_embedding, embed_full)[0]
    sim_kw = cosine_similarity(new_embedding, embed_kw)[0]

    if similarity_type == "text_embedding":
        result_sim = sim_full
        print("Using text embedding similarity: ", result_sim)
    elif similarity_type == "keywords_embedding":
        result_sim = sim_kw
        print("Using keywords embedding similarity: ", result_sim)
    elif similarity_type == "combined_embedding":
        result_sim = alpha * sim_kw + (1 - alpha) * sim_full
        print("Using combined embedding similarity: ", result_sim)

    df["similarity"] = result_sim
    top_indices = df["similarity"].argsort()[-top_k:][::-1]
    top_results = df.iloc[top_indices][
        ["judul", "source", "link", "similarity", "content", "waktu"]
    ].copy()
    top_results["article_id"] = top_indices  # Add article_id for mapping

    return top_results, top_indices
    # return df.sort_values(by="similarity", ascending=False).head(top_k)[
    #     ["judul", "source", "link", "similarity", "content", "waktu"]
    # ]


# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query = ""
    top_k = 5
    similarity_type = "combined_embedding"
    response_time = None

    if request.method == "POST":
        query = request.form.get("query", "")
        top_k = int(request.form.get("top_k", 5))
        similarity_type = request.form.get("similarity_type", "combined_embedding")

        if query.strip():
            start_time = time.perf_counter()
            results, _ = get_top_matches(
                query, df.copy(), model, top_k=top_k, similarity_type=similarity_type
            )
            end_time = time.perf_counter()
            results["waktu"] = pd.to_datetime(results["waktu"])
            results["waktu"] = results["waktu"].dt.strftime("%d-%m-%Y")
            response_time = end_time - start_time

    return render_template(
        "index.html",
        results=results,
        query=query,
        top_k=top_k,
        similarity_type=similarity_type,
        response_time=response_time,
    )


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


@app.route("/evaluate")
def evaluate():
    with open("annotated_data_new.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    df_annot = pd.DataFrame(data)
    df_annot = df_annot[df_annot["relevance_score"].notnull()]
    df_annot["relevance_score"] = df_annot["relevance_score"].astype(int)
    df_annot["query_length"] = df_annot["query"].apply(classify_query_length)

    grouped = df_annot.groupby(["query", "method", "query_length"])
    metrics_by_length = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (query, method, length_category), group in grouped:
        relevance_scores = group.sort_values("rank")["relevance_score"].tolist()
        padded_scores = (
            relevance_scores + [0] * (5 - len(relevance_scores))
            if len(relevance_scores) < 5
            else relevance_scores[:5]
        )
        ideal_scores = [sorted(padded_scores, reverse=True)]
        predicted_scores = [padded_scores]

        # Metrics
        ndcg_at_5 = ndcg_score(ideal_scores, predicted_scores, k=5)
        average_precision = apk(relevance_scores, k=5)

        # Precision@5: proportion of relevant results in top-5
        precision_at_5 = sum([1 for score in relevance_scores[:5] if score > 0]) / 5

        # Recall@5: relevant retrieved in top-5 / total relevant items
        total_relevant_items = sum([1 for score in relevance_scores if score > 0])
        recall_at_5 = (
            sum([1 for score in relevance_scores[:5] if score > 0])
            / total_relevant_items
            if total_relevant_items > 0
            else 0
        )
        # Logging for debug
        # print("===")
        # print(f"Query: {query}")
        # print(f"Method: {method}")
        # print(f"Query Length Category: {length_category}")
        # print(f"Relevance Scores (raw): {relevance_scores}")
        # print(f"Padded Scores: {padded_scores}")
        # print(f"Ideal Scores: {ideal_scores}")
        # print(f"Predicted Scores: {predicted_scores}")
        # print(f"nDCG@5: {ndcg_at_5:.4f}")
        # print(f"MAP@5: {average_precision:.4f}")
        # print(f"Precision@5: {precision_at_5:.4f}")
        # print(f"Recall@5: {recall_at_5:.4f}")
        # print("===")

        metrics_by_length[method][length_category]["nDCG@5"].append(ndcg_at_5)
        metrics_by_length[method][length_category]["MAP@5"].append(average_precision)
        metrics_by_length[method][length_category]["P@5"].append(precision_at_5)
        metrics_by_length[method][length_category]["R@5"].append(recall_at_5)

    summary = []
    chart_labels = []
    metric_names = ["nDCG@5", "MAP@5", "Precision@5", "Recall@5"]
    chart_data = {m: [] for m in metric_names}
    methods_set = set()

    for method, length_dict in metrics_by_length.items():
        for length_category, vals in length_dict.items():
            methods_set.add(method)
            label = f"{method} ({length_category})"
            chart_labels.append(label)

            summary.append(
                {
                    "Method": method,
                    "Query Length": length_category,
                    "nDCG@5": np.mean(vals["nDCG@5"]),
                    "MAP@5": np.mean(vals["MAP@5"]),
                    "Precision@5": np.mean(vals["P@5"]),
                    "Recall@5": np.mean(vals["R@5"]),
                }
            )

            chart_data["nDCG@5"].append(np.mean(vals["nDCG@5"]))
            chart_data["MAP@5"].append(np.mean(vals["MAP@5"]))
            chart_data["Precision@5"].append(np.mean(vals["P@5"]))
            chart_data["Recall@5"].append(np.mean(vals["R@5"]))

    summary_df = pd.DataFrame(summary)

    # Prepare datasets for chart.js
    datasets = []
    colors = {
        "nDCG@5": "#2a4d69",
        "MAP@5": "#98c1d9",
        "Precision@5": "#e0fbfc",
        "Recall@5": "#ee6c4d",
    }
    for metric in metric_names:
        m = metric.split("@")[0]
        datasets.append(
            {
                "label": m,
                "data": chart_data[metric],
                "backgroundColor": colors[metric],
            }
        )

    chart_payload = {"labels": chart_labels, "datasets": datasets}

    return render_template(
        "evaluate.html", table=summary_df.to_html(index=False), chart_data=chart_payload
    )


if __name__ == "__main__":
    app.run(debug=True)
