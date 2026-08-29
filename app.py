import streamlit as st
import pickle
import re


# ========================================
# 1. Load Model and TF-IDF Vectorizer
# ========================================

with open("sentiment_nlp_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


# ========================================
# 2. Stopwords
# ========================================

stopword_text = """
a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once
here there all any both each few more most other some such only own same so than too very
is am are was were be been being have has had having do does did doing i me my myself we our
ours you your yours he him his she her hers it its they them their what which who whom this
that these those will would shall should can could may might must
"""

stopwords = set(stopword_text.split())


# ========================================
# 3. Clean Text
# ========================================

def clean_text(text):

    # Make text lowercase
    text = text.lower()

    # Remove links
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove stopwords and very short words
    words = text.split()
    words = [
        word
        for word in words
        if word not in stopwords and len(word) > 2
    ]

    return " ".join(words)


# ========================================
# 4. Streamlit Page
# ========================================

st.set_page_config(
    page_title="Twitter Sentiment Predictor",
    page_icon="💬",
    layout="centered"
)

st.title("Twitter Sentiment Predictor")

st.write(
    "Enter a tweet or short sentence and the model will predict its sentiment."
)

user_text = st.text_area(
    "Enter text:",
    placeholder="Example: I really enjoyed playing this game!",
    height=150
)


# ========================================
# 5. Prediction
# ========================================

if st.button("Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Clean the text
        cleaned_text = clean_text(user_text)

        if cleaned_text == "":
            st.warning(
                "There is not enough useful text left after cleaning."
            )

        else:

            # Convert text to TF-IDF features
            text_tfidf = tfidf.transform([cleaned_text])

            # Predict
            prediction = model.predict(text_tfidf)[0]

            # Show result
            st.subheader("Prediction")

            if prediction == "Positive":
                st.success("Positive")

            elif prediction == "Negative":
                st.error("Negative")

            elif prediction == "Neutral":
                st.info("Neutral")

            else:
                st.warning("Irrelevant")

            # Optional: show cleaned text
            with st.expander("Show cleaned text"):
                st.write(cleaned_text)
