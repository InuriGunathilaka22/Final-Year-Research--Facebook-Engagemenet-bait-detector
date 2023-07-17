import streamlit as st
import pandas as pd
import re
import spacy
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from PIL import Image

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Step 1: Load the data
# Read the Excel file
df = pd.read_excel('labeled_data.xlsx')

# Save as CSV file
df.to_csv('dataset.csv', index=False)

# ...

def extract_keywords(caption, engagement_keywords):
    doc = nlp(caption)
    keywords = []
    if pd.notna(engagement_keywords) and engagement_keywords is not None:
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ not in ["PRON", "DET"]:
                keywords.append(chunk.lemma_.lower())
        keywords += engagement_keywords
    return keywords

# ...

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Step 2: Preprocessing
# Text cleaning
df["Cleaned Captions"] = df["Facebook Captions"].apply(lambda caption: re.sub(r"[^a-zA-Z]", " ", caption))

# Convert text to lowercase
df["Cleaned Captions"] = df["Cleaned Captions"].str.lower()

# Tokenization
df["Tokens"] = df["Cleaned Captions"].apply(lambda caption: word_tokenize(caption))

# Stopword removal
stop_words = set(stopwords.words("english"))
df["Tokens"] = df["Tokens"].apply(lambda tokens: [token for token in tokens if token not in stop_words])

# Map labels from "0" to "non-engagement bait" and "1" to "engagement bait"
df["Label"] = df["Label"].map({0: "non-engagement bait", 1: "engagement bait"})

# Fit the vectorizer on the tokenized captions
corpus = df["Tokens"].apply(lambda tokens: " ".join(tokens))
vectorizer.fit(corpus)

# Split the data into features (X) and labels (y)
X = vectorizer.transform(corpus)
y = df["Label"]

# Initialize the classifier
classifier = LinearSVC()

# Fit the classifier with all the data
classifier.fit(X, y)

# ...

def classify_caption(caption, stop_words):
    cleaned_caption = re.sub(r"[^a-zA-Z]", " ", caption).lower()  # Clean and convert to lowercase
    tokens = word_tokenize(cleaned_caption)  # Tokenize
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    keywords = extract_keywords(cleaned_caption, None)  # Extract keywords without matching rows
    tokens += keywords  # Combine tokens and keywords
    caption_vector = vectorizer.transform([" ".join(tokens)])  # Vectorize
    prediction = classifier.predict(caption_vector)[0]  # Make prediction
    return prediction, keywords

# ...

# Create the UI
st.set_page_config(page_title="Engagement-bait Detector")

# Set app title and description
st.markdown("# 🧐 Engagement-bait Detector")
st.markdown("This application is a Streamlit dashboard that can be used to predict whether an article caption/headline is engagement-bait or not. Simply submit the caption/headline that you would like to test below:")

# Get user input
caption = st.text_input("Enter a caption:")

# Classify caption
if caption:
    prediction, keywords = classify_caption(caption, stop_words)
    if prediction == "engagement bait":
        st.error('This headline is engagement-bait')
        if keywords:
            st.markdown("### Identified Keywords:")
            st.write(", ".join(keywords))
    else:
        st.success('This is non-engagement bait')
        st.balloons()

# Display image
image = Image.open('Engagement_Baiting.png')
st.image(image, width=660, caption='Source: https://cdn.pixabay.com/photo/2016/11/22/23/40/electronics-1851218_960_720.jpg')
