# COVID-19 Tweets: Topic Modeling & Sentiment Analysis Pipeline

An end-to-end Text Mining and Natural Language Processing (NLP) pipeline implemented in **R**. This project analyzes a large-scale dataset of **170k+ tweets** regarding the COVID-19 pandemic to uncover latent discussion themes and map the global emotional climate during the crisis.

---

## 📊 Project Overview & Architecture
The project lifecycle follows a structured text mining pipeline:
1. **Data Collection & Filtering**: Handling and cleaning 179,108 raw tweets down to 169,495 valid records.
2. **Text Preprocessing**: Lowercasing, tokenization, stopword removal, custom filtering, and Porter Stemming using the `tm` package.
3. **Vectorization**: Generating a Document-Term Matrix (DTM) with Sparsity Control (1%).
4. **Exploratory Text Analysis**: Statistical frequency mapping and global vocabulary visualizations.
5. **Topic Modeling (LDA)**: Unsupervised clustering using Latent Dirichlet Allocation, optimized via Perplexity and Coherence metrics.
6. **Sentiment Analysis**: Double-layered lexicon extraction using the `syuzhet` framework and the `NRC` emotion lexicon.

---

## 📈 Key Visualizations & Insights

### 1. Overall Sentiment Distribution
Using a lexicon-based approach with a designated **neutrality zone ($\pm0.15$)**, the overall sentiment orientation highlights a predominantly optimistic outlook, despite the gravity of the pandemic.

<p align="center">
  <img src="screenshots/4.1. SentimentPieChart.png" width="500" alt="Overall Sentiment Distribution">
</p>

### 2. Emotional Profile (NRC Lexicon)
Going beyond binary polarity, the multi-categorical emotional analysis tracks 8 basic psychological states. **Trust** emerges as the leading factor, indicating public reliance on healthcare directives, followed closely by **Fear**.

<p align="center">
  <img src="screenshots/4.2. NRCEmotionsBarChart.png" width="600" alt="NRC Emotions Profile">
</p>

### 3. Latent Topics & Sentiment Matching
The LDA model successfully identified $k=5$ distinct thematic clusters. Mapping sentiment weights across these clusters revealed key public dynamics:
* **Prevention & Masks**: Highest volume of tweets with overwhelming positive support for protective measures.
* **Politics & Public Opinion**: Triggered the highest ratio of negative sentiment, highlighting strong political polarization (prominent keywords: *Trump*, *realdonaldtrump*).
* **Science & Vaccines**: The most controversial and closely contested topic, where high optimism for clinical breakthroughs was strictly balanced by public skepticism and vaccine hesitancy.

<p align="center">
  <img src="screenshots/5.1 groupedBarChart.png" width="650" alt="Sentiment Counts per Topic">
</p>

---

## 🛠️ Tech Stack & Libraries
* **Language**: R
* **Text Mining Framework**: `tm`, `SnowballC`
* **Topic Modeling**: `topicmodels`
* **Sentiment Analysis**: `syuzhet`
* **Data Visualization**: `ggplot2`, `wordcloud`, `RColorBrewer`

## 📂 Dataset Source
The data utilized in this study is based on the Kaggle COVID-19 Tweets dataset (https://www.kaggle.com/datasets/gpreda/covid19-tweets). Due to GitHub file size limitations, the raw data is not hosted here.

---
*Developed as part of the Data Warehouses and Data Mining Course - Department of Digital Systems, University of Thessaly.*
