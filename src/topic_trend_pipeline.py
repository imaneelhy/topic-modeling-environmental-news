!pip install beautifulsoup4 lxml -q

import os
import re
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

RANDOM_SEED = 42

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "news_sample.csv")

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_TOPICS = 8
N_TOP_WORDS = 12
TIME_FREQ = "M"   # month


# ---------------------------------------------------------------------
# 0. Download & standardize real environmental news dataset
# ---------------------------------------------------------------------

def download_environmental_news(limit=1000) -> pd.DataFrame:
    """
    Download environmental news from the Taiwan Ministry of Environment
    open-data API (real news articles).

    We keep: id, date, source, title, text.
    """
    url = (
        "https://data.moenv.gov.tw/api/v2/mnews_p_01"
        f"?api_key=b7df779e-71a6-4148-8379-5afbd441d803"
        f"&format=CSV&limit={limit}&sort=ImportDate+desc"
    )
    df_raw = pd.read_csv(url)
    print("[INFO] Downloaded raw dataset from API.")
    print("       Raw shape:", df_raw.shape)
    print("       Raw columns:", df_raw.columns.tolist())

    df_std = pd.DataFrame()
    df_std["id"] = df_raw["newsno"]
    df_std["date"] = df_raw["newsdate"]
    df_std["source"] = df_raw["newssource"]
    df_std["title"] = df_raw["newstitle"]
    df_std["text"] = df_raw["newscontent"]

    df_std = df_std.dropna(subset=["date", "text"]).reset_index(drop=True)

    print("[INFO] Standardized dataset shape:", df_std.shape)
    return df_std


# ---------------------------------------------------------------------
# 1. Utilities (reproducibility, NLTK, cleaning)
# ---------------------------------------------------------------------

def set_random_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)


def setup_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def html_to_text(raw_html: str) -> str:
    """
    Use BeautifulSoup to strip HTML tags and styles.
    """
    try:
        soup = BeautifulSoup(str(raw_html), "lxml")
        return soup.get_text(separator=" ")
    except Exception:
        return str(raw_html)


def basic_clean(text: str) -> str:
    """
    HTML stripping + light normalization:
    - strip HTML tags
    - lowercase
    - remove URLs
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = html_to_text(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(basic_clean)
    df["clean_len"] = df["clean_text"].str.split().apply(len)
    # drop extremely short documents
    df = df[df["clean_len"] >= 5].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 2. Stopwords (English + Chinese domain + boilerplate)
# ---------------------------------------------------------------------

def build_stopwords():
    english_stop = stopwords.words("english")

    # Generic news / government boilerplate (English)
    domain_en = [
        "taiwan", "ministry", "environment", "environmental",
        "news", "said", "say", "says",
        "government", "official", "authorities",
        "email", "tel", "fax",
    ]

    # High-frequency Chinese boilerplate phrases from your results
    # (you can keep extending this list after inspecting vocab_stats.csv)
    domain_zh = [
        "環境部", "環境部表示", "環境部強調", "環境部說明", "環境部今", "環境部指出",
        "環境部重申", "環境部提醒", "環境部修正發布", "環境部預告修正",
        "環境即時通",
        "日舉辦", "辦理", "活動", "本次", "本部", "以上",
        "此外", "另外", "因此", "其中", "同時",
        "網址", "下載", "查詢", "請參閱", "附加檔案",
        "gov", "tw", "moenv",
        # very generic numeric tokens that show up a lot
        "113", "114", "000", "10", "12", "29",
    ]

    # Residual markup / formatting tokens as safety net
    markup_stop = [
        "span", "style", "margin", "text", "font", "background",
        "border", "color", "align", "justify", "layout", "grid",
        "class", "href", "br", "nbsp",
        "cm", "pt", "px", "0cm", "0pt"
    ]

    return english_stop + domain_en + domain_zh + markup_stop


# ---------------------------------------------------------------------
# 3. Topic modelling helpers (LDA / NMF)
# ---------------------------------------------------------------------

def fit_lda(docs, n_topics=10, max_features=8000):
    sw = build_stopwords()
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=sw,
        min_df=5,       # ignore very rare terms
        max_df=0.7,     # ignore terms that appear in >70% of docs
    )
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=RANDOM_SEED,
    )
    doc_topic = lda.fit_transform(X)
    return lda, vectorizer, doc_topic


def fit_nmf(docs, n_topics=10, max_features=8000):
    sw = build_stopwords()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=sw,
        min_df=5,
        max_df=0.7,
    )
    X = vectorizer.fit_transform(docs)

    nmf = NMF(
        n_components=n_topics,
        init="nndsvd",
        random_state=RANDOM_SEED,
        max_iter=400,
    )
    doc_topic = nmf.fit_transform(X)
    return nmf, vectorizer, doc_topic


def print_topics(model, feature_names, n_top_words=10, header="LDA"):
    lines = [f"=== {header} topics ==="]
    for topic_idx, topic in enumerate(model.components_):
        top_indices = np.argsort(topic)[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        lines.append(f"Topic {topic_idx:02d}: " + ", ".join(top_words))
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------
# 4. Vocabulary diagnostics (for stopword curation)
# ---------------------------------------------------------------------

def dump_vocabulary_stats(docs, out_path):
    """
    Build a simple bag-of-words model and dump token frequencies.
    You can open vocab_stats.csv and decide which tokens to add
    to stopwords. This is very 'academic'.
    """
    sw = build_stopwords()
    vec = CountVectorizer(
        stop_words=sw,
        min_df=5,
        max_df=0.7,
    )
    X = vec.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    df_vocab = pd.DataFrame({"token": vocab, "count": freqs})
    df_vocab = df_vocab.sort_values("count", ascending=False)
    df_vocab.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved vocabulary statistics to {out_path}")


# ---------------------------------------------------------------------
# 5. Time binning, trend analysis, qualitative examples
# ---------------------------------------------------------------------

def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def add_time_bins(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = df.copy()
    df["date"] = parse_date(df["date"])
    df = df.dropna(subset=["date"])
    df["time_bin"] = df["date"].dt.to_period(freq).dt.to_timestamp()
    return df


def compute_topic_trends(df: pd.DataFrame, doc_topic: np.ndarray, model_name="lda"):
    df = df.copy()
    df[f"{model_name}_topic"] = doc_topic.argmax(axis=1)

    counts = (
        df.groupby(["time_bin", f"{model_name}_topic"])
          .size()
          .reset_index(name="count")
    )
    total_per_time = counts.groupby("time_bin")["count"].transform("sum")
    counts["proportion"] = counts["count"] / total_per_time
    return df, counts


def plot_topic_trends(counts: pd.DataFrame, model_name="lda"):
    if counts.empty:
        print(f"[WARN] No counts for {model_name}, skipping plot.")
        return

    pivot = counts.pivot_table(
        index="time_bin",
        columns=f"{model_name}_topic",
        values="proportion",
        fill_value=0.0,
    )

    plt.figure(figsize=(10, 5))
    for topic_id in pivot.columns:
        plt.plot(pivot.index, pivot[topic_id], marker="o", label=f"Topic {topic_id}")
    plt.xlabel("Time")
    plt.ylabel("Topic frequency (proportion of articles)")
    plt.title(f"{model_name.upper()} topic trends over time")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"{model_name}_topic_trends.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[INFO] Saved {model_name} topic trend plot to {fig_path}")


def save_example_titles_per_topic(df, topic_col, model_name, k=5):
    lines = []
    for topic_id in sorted(df[topic_col].unique()):
        subset = df[df[topic_col] == topic_id].sort_values("date")
        lines.append(f"=== {model_name.upper()} Topic {topic_id} ===")
        for _, row in subset.head(k).iterrows():
            date_str = str(row["date"])[:10]
            title = str(row["title"])
            source = str(row["source"])
            lines.append(f"{date_str} | {source} | {title}")
        lines.append("")
    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, f"{model_name}_example_titles.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Saved example titles per topic to {path}")


def summarize_topic_distribution(df, topic_col, model_name):
    counts = df[topic_col].value_counts().sort_index()
    lines = [f"=== {model_name.upper()} topic document counts ==="]
    total = counts.sum()
    for tid, c in counts.items():
        lines.append(f"Topic {tid}: {c} docs ({c/total:.2%})")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------

def main():
    set_random_seed(RANDOM_SEED)
    setup_nltk()

    print("[INFO] Downloading real environmental news dataset and overwriting data/news_sample.csv ...")
    df_std = download_environmental_news(limit=1000)
    df_std.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Wrote standardized dataset to {DATA_PATH}")

    df = df_std.copy()
    print(f"[INFO] Loaded {len(df)} articles from {DATA_PATH}")

    df = preprocess_texts(df)
    print(f"[INFO] After cleaning, {len(df)} articles remain.")
    df = add_time_bins(df, freq=TIME_FREQ)
    print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")

    # Title + body
    docs = (df["title"].fillna("") + " " + df["clean_text"]).tolist()

    # --- Vocabulary diagnostics (for the report) ---
    dump_vocabulary_stats(docs, os.path.join(RESULTS_DIR, "vocab_stats.csv"))

    # ---------------- LDA ----------------
    print("\n[INFO] Fitting LDA...")
    lda_model, lda_vec, lda_doc_topic = fit_lda(
        docs, n_topics=N_TOPICS, max_features=8000
    )
    lda_feature_names = lda_vec.get_feature_names_out()
    lda_topics_text = print_topics(
        lda_model, lda_feature_names, n_top_words=N_TOP_WORDS, header="LDA"
    )
    with open(os.path.join(RESULTS_DIR, "lda_topics.txt"), "w", encoding="utf-8") as f:
        f.write(lda_topics_text)

    lda_confidence = lda_doc_topic.max(axis=1).mean()
    print(f"[INFO] Mean LDA dominant-topic probability: {lda_confidence:.3f}")

    df_lda, lda_counts = compute_topic_trends(df, lda_doc_topic, model_name="lda")
    lda_counts.to_csv(os.path.join(RESULTS_DIR, "lda_topic_trends.csv"), index=False)
    plot_topic_trends(lda_counts, model_name="lda")
    save_example_titles_per_topic(df_lda, "lda_topic", "lda", k=5)

    # ---------------- NMF ----------------
    print("\n[INFO] Fitting NMF...")
    nmf_model, nmf_vec, nmf_doc_topic = fit_nmf(
        docs, n_topics=N_TOPICS, max_features=8000
    )
    nmf_feature_names = nmf_vec.get_feature_names_out()
    nmf_topics_text = print_topics(
        nmf_model, nmf_feature_names, n_top_words=N_TOP_WORDS, header="NMF"
    )
    with open(os.path.join(RESULTS_DIR, "nmf_topics.txt"), "w", encoding="utf-8") as f:
        f.write(nmf_topics_text)

    nmf_confidence = nmf_doc_topic.max(axis=1).mean()
    print(f"[INFO] Mean NMF dominant-topic probability: {nmf_confidence:.3f}")

    df_nmf, nmf_counts = compute_topic_trends(df, nmf_doc_topic, model_name="nmf")
    nmf_counts.to_csv(os.path.join(RESULTS_DIR, "nmf_topic_trends.csv"), index=False)
    plot_topic_trends(nmf_counts, model_name="nmf")
    save_example_titles_per_topic(df_nmf, "nmf_topic", "nmf", k=5)

    # ---------------- Summary ----------------
    summary_lines = []
    summary_lines.append("=== Dataset summary ===")
    summary_lines.append(f"Num articles (after cleaning): {len(df)}")
    summary_lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    summary_lines.append("")
    summary_lines.append(f"LDA mean dominant-topic prob: {lda_confidence:.3f}")
    summary_lines.append(f"NMF mean dominant-topic prob: {nmf_confidence:.3f}")
    summary_lines.append("")
    summary_lines.append(summarize_topic_distribution(df_lda, "lda_topic", "lda"))
    summary_lines.append("")
    summary_lines.append(summarize_topic_distribution(df_nmf, "nmf_topic", "nmf"))

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print("\n[INFO] Done. Check the 'results' folder for all outputs.")

    try:
        from IPython.display import Image, display
        display(Image(os.path.join(FIGURES_DIR, "lda_topic_trends.png")))
    except Exception:
        pass


# Run the full pipeline
main()
