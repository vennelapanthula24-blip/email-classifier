import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page settings
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧"
)

st.title("📧 Spam Email Classifier")
st.write("Enter a message below and the AI model will predict whether it is spam or not.")

# Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels to numbers
data["label_num"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Prepare data
X = data["message"]
y = data["label_num"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_tfidf, y)

# User input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize!"
)

if st.button("Check Message"):
    if message.strip():
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
    else:
        st.warning("Please enter a message.")