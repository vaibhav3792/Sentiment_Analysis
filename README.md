# 🎬 IMDB Movie Review Sentiment Analysis

An End-to-End Deep Learning project that classifies movie reviews as **Positive** or **Negative** using an LSTM Neural Network.

## 🔗 Live Demo
👉 **[Click here to test the App](https://sentiment-app-live.streamlit.app/)**

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** TensorFlow / Keras
* **Model Architecture:** LSTM (Long Short-Term Memory) with Embedding Layer
* **Training Data:** IMDB Dataset (25,000 reviews)

## 🚀 How to Run Locally
1.  Clone the repository:
    ```bash
    git clone [https://github.com/vaibhav3792/Sentiment_Analysis.git](https://github.com/vaibhav3792/Sentiment_Analysis.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run main.py
    ```

## 📊 Model Performance
* **Accuracy:** ~85% on validation data.
* **Key Challenge:** Overcame the "Vanishing Gradient" problem of Simple RNNs by implementing LSTM.