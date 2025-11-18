

````markdown
# Strategic Topic & Trend Analysis of Environmental News

> Can classical topic modelling + trend analysis extract useful “strategic intelligence” from real-world environmental news?

This project builds an end-to-end pipeline that:

- **Collects** several years of environmental news from the Taiwan Ministry of Environment open-data API.
- **Cleans** and normalizes the text (HTML removal, bilingual stopwords, etc.).
- Fits two **topic models** (LDA & NMF).
- Tracks how each topic’s **frequency evolves over time**.
- Saves topic word lists, example articles, and time-series plots for qualitative analysis.

The project is designed to look and feel like a small research study, aimed at internships in:

- Graph / NLP for strategic intelligence  
- Topic / trend analysis on news streams  
- Risk & information management

---

## 1. Research Question

> **RQ:** Can topic modelling (LDA / NMF) combined with simple time-series analysis reveal meaningful strategic themes and their evolution in multi-year environmental news?

Concretely, we want to detect themes such as:

- air quality warnings and public health advice,  
- net-zero & carbon-fee policy,  
- PFAS / hazardous chemicals and resource-circulation law,  
- city-level climate initiatives,  
- climate commitments (Paris, NDCs),  
- environmental campaigns (e.g., cigarette-butt reduction, “green table” food).

…and see how often these appear over time.

---

## 2. Repository Structure

```text
.
├── data/
│   └── news_sample.csv          # standardized news dataset (auto-created)
├── results/
│   ├── figures/
│   │   ├── lda_topic_trends.png
│   │   └── nmf_topic_trends.png
│   ├── lda_topics.txt           # top words per LDA topic
│   ├── nmf_topics.txt           # top words per NMF topic
│   ├── lda_example_titles.txt   # representative headlines per LDA topic
│   ├── nmf_example_titles.txt   # representative headlines per NMF topic
│   ├── lda_topic_trends.csv     # topic proportions per month (LDA)
│   ├── nmf_topic_trends.csv     # topic proportions per month (NMF)
│   ├── vocab_stats.csv          # token frequency statistics
│   └── summary.txt              # high-level metrics and topic counts
├── src/
│   └── topic_trend_pipeline.py  # main script (download + train + plots)
└── README.md
````

> **Note:** `data/news_sample.csv` is created automatically by the script by calling the public API.

---

## 3. Data

### Source

News articles from the **Taiwan Ministry of Environment** open-data API:

```text
Dataset: mnews_p_01 (environmental news)
Format: CSV, ~1000 latest articles
Fields: newsno, newstitle, newscontent, newssource, newsdate, ...
```

The script:

1. downloads up to 1000 news items,

2. standardises them into:

   * `id`
   * `date`
   * `source`
   * `title`
   * `text`

3. saves a local copy to `data/news_sample.csv` for reproducibility.

### Final corpus

After cleaning:

* **996** raw articles
* **795** kept after filtering (very short / empty texts removed)
* **Time span:** 2020-01-07 to 2025-11-18

Language is primarily **Traditional Chinese**, with some English technical terms.

---

## 4. Methods

### 4.1 Preprocessing

Steps:

1. **HTML removal**

   * Use `BeautifulSoup` to convert `newscontent` into plain text.

2. **Text normalization**

   * lowercasing
   * remove URLs
   * collapse whitespace
   * drop documents with < 5 tokens

3. **Bilingual stopwords**

   * English stopwords from NLTK.

   * Extra English boilerplate: `taiwan`, `ministry`, `environment`, `environmental`, `news`, `said`, `government`, etc.

   * Chinese boilerplate phrases inspected from token statistics:

     * 「環境部表示」「環境部強調」「環境部說明」「環境部今」
     * 「此外」「另外」「因此」「其中」「同時」
     * 「網址」「下載」「請參閱」「附加檔案」
     * and frequent numeric tokens like `113`, `114`, `000`, `10`…

   * Residual HTML/formatting tokens: `span`, `style`, `margin`, `font`, `px`, `pt`, etc.

4. **Vocabulary diagnostics**

   Before topic modelling, we build a simple bag-of-words model and export
   `results/vocab_stats.csv` (token, count).
   This file is used to iteratively refine the domain-specific stopword list.

5. **Vectorization**

   * For **LDA**: `CountVectorizer` with

     * `min_df = 5` (ignore tokens in < 5 docs)
     * `max_df = 0.7` (ignore tokens in > 70% of docs)
     * custom stopwords list.
   * For **NMF**: `TfidfVectorizer` with the same `min_df`, `max_df`, and stopwords.

### 4.2 Topic Models

We fit two models, both with **8 topics**:

1. **Latent Dirichlet Allocation (LDA)**

   * Input: word counts (CountVectorizer)
   * `n_components=8`, `learning_method="batch"`, `random_state=42`.

2. **Non-negative Matrix Factorization (NMF)**

   * Input: TF–IDF features
   * `n_components=8`, `init="nndsvd"`, `max_iter=400`, `random_state=42`.

For each document, we record the **dominant topic** (argmax over topic distribution).

### 4.3 Time-series / Trend Analysis

* Convert article dates to monthly bins (`time_bin` via `dt.to_period('M')`).

* For each month and topic, compute:

  [
  \text{topic_proportion}(t, k) =
  \frac{\text{# documents in month } t \text{ whose dominant topic } = k}
  {\text{# documents in month } t}
  ]

* Save results:

  * `results/lda_topic_trends.csv`
  * `results/nmf_topic_trends.csv`

* Plot **topic proportion vs. time** for each topic:

  * `results/figures/lda_topic_trends.png`
  * `results/figures/nmf_topic_trends.png`

### 4.4 Qualitative inspection

For interpretability:

* `lda_topics.txt` / `nmf_topics.txt`: top N words per topic.
* `lda_example_titles.txt` / `nmf_example_titles.txt`:
  for each topic, up to 5 representative article titles (with date & source).

---

## 5. How to Run

### 5.1 Environment

Python 3.10+ recommended.

Dependencies:

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `nltk`
* `beautifulsoup4`
* `lxml`

Install (example):

```bash
pip install numpy pandas scikit-learn matplotlib nltk beautifulsoup4 lxml
```

On first run, `nltk` will download the English stopwords corpus if missing.

### 5.2 Run the pipeline

From the project root:

```bash
python src/topic_trend_pipeline.py
```

The script will:

1. Download the latest environmental news CSV from the API.
2. Standardize and save it to `data/news_sample.csv`.
3. Clean & preprocess the text.
4. Fit **LDA** and **NMF**.
5. Export all topics, examples, trends, and summary files to `results/`.

Open:

* `results/figures/lda_topic_trends.png` – main trend plot used in the analysis.
* `results/summary.txt` – high-level metrics and topic counts.
* `results/lda_topics.txt` – top words per topic.

---

## 6. Results

### 6.1 LDA quality

* Number of documents after cleaning: **795**

* Number of topics: **8**

* **Mean dominant-topic probability** (how confident the model is per doc):

  * LDA: **0.70** (reasonably sharp topics)
  * NMF: **0.11** (very overlapping; less useful for monitoring)

* **Document share per LDA topic:**

  | Topic | Share of docs |
  | ----- | ------------- |
  | 0     | 6.5 %         |
  | 1     | 14.2 %        |
  | 2     | 15.9 %        |
  | 3     | 15.9 %        |
  | 4     | 5.2 %         |
  | 5     | 15.6 %        |
  | 6     | 12.2 %        |
  | 7     | 14.6 %        |

No single topic dominates the corpus; each captures a coherent subset of news.

### 6.2 Interpreted LDA topics

After inspecting top words and example headlines, we assign the following labels:

| ID | Label (informal)                                           | Brief description                                                                                                                     |
| -- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 0  | **Anti-litter / Cigarette-butt campaigns**                 | Campaigns like “菸蒂不落地”, youth-focused clean-environment actions, cigarette-butt reduction.                                            |
| 1  | **Solid-recovered fuel & waste-law revisions**             | Drafts and amendments around 廢棄物清理法, SFR (固體再生燃料), pollutant thresholds (PM, mg).                                                     |
| 2  | **City-level climate initiatives & transparency**          | City-specific reports (Taichung, Taoyuan, Kaohsiung, New Taipei, etc.), climate information portals, green-skills employment reports. |
| 3  | **Net-zero skills & carbon-fee policy**                    | Net-zero “green talent” training courses, 淨零綠生活 campaigns, 碳費收費辦法, self-mitigation plan regulations.                                  |
| 4  | **Air-quality warnings & health guidance**                 | Air-quality alerts (立方公尺, PM), health advice (reduce outdoor activity, wear masks), mobile-app alerts.                                |
| 5  | **PFAS / hazardous substances & resource-circulation law** | News about PFAS regulation, 資源循環推動法, long-term circular-economy measures, occasionally AI-related initiatives.                        |
| 6  | **Climate commitments, NDCs & marine campaigns**           | References to the Paris Agreement, NDCs, 氣候變遷因應法, “向海致敬” campaigns, and environmental education committees.                           |
| 7  | **Cultural events & green lifestyle campaigns**            | Initiatives like 新紙錢三燒 (cleaner temple burning), 綠食飯桌 (green diet), and other themed campaigns by the Resource Circulation agency.    |

The text files `results/lda_example_titles.txt` give concrete article titles for each topic.

### 6.3 Topic trends over time

The main figure (`results/figures/lda_topic_trends.png`) shows topic frequencies per month from 2020–2025.

Some qualitative observations:

* **Early years (2020–2021)**

  * Very few articles per month → topic proportions are spiky (single topics can dominate a month).
* **Net-zero & carbon fee (Topic 3)**

  * Almost absent before 2022.
  * Noticeable growth from 2023 onward, in line with draft carbon-fee regulations and net-zero training programs.
* **Air-quality warnings (Topic 4)**

  * Recurrent peaks, especially in months associated with poorer air quality.
  * These topics often include operational guidance on mask-wearing and avoiding outdoor activity.
* **PFAS / hazardous chemical regulation (Topic 5)**

  * Appears later in the timeline, reflecting newer regulatory focus on PFAS and related resource-circulation reforms.
* **Cultural / green-lifestyle campaigns (Topic 0 & 7)**

  * Associated with specific campaigns and festivals (e.g., New Year, religious events).
  * Their spikes can signal periods when environmental messaging targets behaviour change.

### 6.4 NMF comparison

NMF topics are more overlapping:

* Mean dominant-topic probability ≈ **0.11**.
* One factor covers ~**54%** of documents, acting as a generic “boilerplate” topic.
* Other NMF topics still map to meaningful patterns (air-quality warnings, net-zero skills, regulatory notices), but are less crisp than LDA.

**Interpretation:** for this dataset, **LDA** provides sharper, more interpretable topics and is preferred for strategic monitoring; **NMF** mainly serves as a comparison baseline.

---

## 7. Limitations

This project also documents its limitations:

* **Language preprocessing**

  * Text is primarily Traditional Chinese.
  * Tokenisation is based on simple whitespace / punctuation; a more robust version would use a dedicated Chinese segmenter or transformer-based embeddings.

* **Model family**

  * Only classical bag-of-words topic models (LDA, NMF) are used.
  * No transformer-based models (e.g. BERTopic) are explored yet.

* **Boilerplate remains**

  * Despite an extended stopword list, some ministry phrases and numbers still appear in topics.
  * This reflects the news style; further cleaning could remove more of this.

* **Single data source**

  * All articles come from one ministry.
  * For a broader strategic-intelligence system, one would integrate multiple news outlets and social media.



---

*Feel free to open an issue or contact me if you have questions or ideas for extensions.*

```
