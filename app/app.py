import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load your pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('baseline_modelVarietydataSet.h5')

model = load_model()

# Load the saved tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

tokenizer = load_tokenizer()

# Set parameters for text preprocessing
MAX_SEQUENCE_LENGTH = 200  # Ensure this matches what the model was trained with

st.title("📰 US Political Fake News Text Detector")
st.write("Detect whether a given piece of news is fake or real based on its content. Enter a URL to analyze its authenticity or test with a sample text.")

# Load and display the image
image = Image.open('list.png')
st.image(image, caption='Source: https://en.wikipedia.org/wiki/List_of_fake_news_websites', use_column_width=True)

# Display clickable links for fake news examples
st.title("🔗 Example Fake News Articles")
st.markdown("[Link 1](https://newsexaminer.net/politics/democratic/trump-democrats-face-different-political-landscape-ahead-of-midterms/)")
st.markdown("[Link 2](https://newsexaminer.net/robert-f-kennedy-jr-suspends-2024-presidential-campaign-endorses-donald-trump/)")
st.markdown("[Link 3](https://newsexaminer.net/trumps-fiery-response-to-harris-dnc-speech-a-social-media-frenzy/)")

# URL input for web scraping
st.title("🔍 Analyze News from a URL")
url = st.text_input("Enter the URL of the news article you want to analyze:")

# Web scraping function to extract text from the URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts, styles, and extract the text content
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n").strip()
        return text[:1000]  # Limit to first 1000 characters
    except requests.exceptions.RequestException as e:
        return f"Error scraping the URL: {e}"

# Text preprocessing function
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_sequences

# Test model with sample text
def test_sample_text():
    sample_text = "The president held a press conference today to discuss the upcoming elections."
    preprocessed_text = preprocess_text(sample_text)
    prediction = model.predict(preprocessed_text)
    fake_prob = prediction[0][0] * 100
    st.write(f"⚠️ Sample Text Potential Fake News Probability: {fake_prob:.2f}%")
    if fake_prob > 50:
        st.write("The sample text is likely Fake.")
    else:
        st.write("The sample text is likely Real.")

# Button to test the model with sample text
if st.button('🧪 Test with Sample Text'):
    test_sample_text()

# Analyze a news article from the given URL
if url:
    with st.spinner("Scraping the text..."):
        scraped_text = scrape_text_from_url(url)
        if "Error" in scraped_text:
            st.error(scraped_text)
        else:
            # Display scraped text
            st.subheader("📄 Scraped Text:")
            st.write(scraped_text)

            # Preprocess and predict
            preprocessed_text = preprocess_text(scraped_text)
            with st.spinner("Analyzing the news article..."):
                prediction = model.predict(preprocessed_text)
                fake_prob = prediction[0][0] * 100

                # Display the prediction
                st.write(f"⚠️ Potential Fake News Probability: {fake_prob:.2f}%")
                if fake_prob > 50:
                    st.write("The news article is likely Fake.")
                else:
                    st.write("The news article is likely Real.")
else:
    st.info("Please enter a URL to start analyzing.")

# Disclaimer
st.write("Disclaimer: This tool uses machine learning algorithms to predict the authenticity of news articles. While it aims to be accurate, it should not be solely relied upon for making critical decisions.")
