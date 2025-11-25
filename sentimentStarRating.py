import json
import pandas as pd
import matplotlib.pyplot as plt


def load_first_n_reviews(filepath, n=1000):
    reviews = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            reviews.append(json.loads(line))
    return pd.DataFrame(reviews)

df = load_first_n_reviews("./data/review.json", n=1000)

print(df.head())
print(df["stars"].value_counts())
df["length"] = df["text"].apply(len)

print("Columns:", df.columns.tolist())
print("\nStar rating distribution:")
print(df["stars"].value_counts().sort_index())

# Add a basic length column (number of characters)
df["length"] = df["text"].astype(str).apply(len)

print("\nBasic statistics for review length:")
print(df["length"].describe())

# Add a Histogram of Review Lengths
plt.figure(figsize=(8,5))
plt.hist(df["length"], bins=40, color="skyblue", edgecolor="black")
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (characters)")
plt.ylabel("Number of Reviews")
plt.show()

# Plot: number of reviews per star rating
star_counts = df["stars"].value_counts().sort_index()

plt.figure()
star_counts.plot(kind="bar")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews per Star Rating")
plt.show()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

analyzer = SentimentIntensityAnalyzer()

df["sentiment"] = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

# Plot: sentiment distribution per star rating
plt.figure(figsize=(8, 5))
sns.boxplot(x="stars", y="sentiment", data=df)
plt.title("Sentiment Score Distribution Across Star Ratings")
plt.show()

# Average sentiment line plot
sent_mean = df.groupby("stars")["sentiment"].mean()
sent_mean.plot(kind="line", marker="o")
plt.title("Average Sentiment by Star Rating")
plt.show()



from wordcloud import WordCloud

def make_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

make_wordcloud(df[df.stars == 1]["text"], "1-Star Review Word Cloud")
make_wordcloud(df[df.stars == 3]["text"], "3-Star Review Word Cloud")
make_wordcloud(df[df.stars == 5]["text"], "5-Star Review Word Cloud")

from sklearn.feature_extraction.text import TfidfVectorizer
# use TF-IDF to sift out comon words

def top_tfidf_words_for_rating(df_in: pd.DataFrame, star: int, n: int = 15):
    subset = df_in[df_in["stars"] == star]["text"].dropna().astype(str).tolist()
    if len(subset) == 0:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(subset)
    scores = X.mean(axis=0).A1
    words = vectorizer.get_feature_names_out()
    tfidf_scores = list(zip(words, scores))
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    return tfidf_scores[:n]

for star in sorted(df["stars"].unique()):
    print(f"\nTop TF-IDF words for {star}-star reviews:")
    top_words = top_tfidf_words_for_rating(df, star, n=10)
    for w, s in top_words:
        print(f"  {w:<20} {s:.4f}")



# Boxplot: review length by star rating - whether negative reviews are better
plt.figure()
data_to_plot = [df[df["stars"] == s]["length"].values for s in sorted(df["stars"].unique())]
plt.boxplot(data_to_plot, labels=sorted(df["stars"].unique()))
plt.xlabel("Star Rating")
plt.ylabel("Review Length (characters)")
plt.title("Review Length Across Star Ratings")
plt.show()



# Scatter: sentiment vs. 'useful' votes (if available)
if "useful" in df.columns:
    plt.figure()
    plt.scatter(df["sentiment"], df["useful"], alpha=0.5)
    plt.xlabel("Sentiment (compound)")
    plt.ylabel("Useful Votes")
    plt.title("Sentiment vs Useful Votes")
    plt.show()
else:
    print("'useful' column not found in this dataset.")