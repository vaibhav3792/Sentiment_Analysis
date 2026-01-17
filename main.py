import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


@st.cache_resource
def load_resources():
    word_index = imdb.get_word_index()
    model = load_model('sentiment_model.keras')
    return word_index, model

word_index, model = load_resources()

# --- HELPER FUNCTION ---
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [1] + [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=100)
    st.header("About this App")
    st.write("This is a Deep Learning project using **LSTM (Long Short-Term Memory)** neural networks.")
    st.write("It was trained on 25,000 movie reviews to understand human sentiment.")
    st.markdown("---")
    st.write("Developed by **You**")


st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Is a review **Positive** or **Negative**? Paste it below to find out.")


user_input = st.text_area("Paste the review here:", height=200, placeholder="Example: The movie was fantastic! I loved the ending.")


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

# When the button is clicked
if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing the vibes..."):
            processed_input = preprocess_text(user_input)
            prediction = model.predict(processed_input)
            score = prediction[0][0]
            sentiment = 'Positive' if score > 0.5 else 'Negative'

        # Display Results
        st.markdown("---")
        st.subheader("Analysis Results")

        
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if sentiment == "Positive":
                st.success(f"**Sentiment:** {sentiment} 😃")
            else:
                st.error(f"**Sentiment:** {sentiment} 😠")

        with result_col2:
            st.metric("Confidence Score", f"{score*100:.2f}%")
            
            # Visual Progress Bar
            st.progress(int(score * 100))

    
        if score > 0.8:
            st.info("💡 The AI is **very confident** this is a good review.")
        elif score < 0.2:
            st.info("💡 The AI is **very confident** this is a bad review.")
        else:
            st.info("💡 The AI is a bit unsure. The language might be mixed.")