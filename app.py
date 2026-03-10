import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Sentimen & Ringkasan AI", page_icon="🤖", layout="wide")
st.title("🤖 Analisis Sentimen & Ringkasan Konteks")
st.write("Aplikasi ini menggunakan **IndoBERT** untuk sentimen dan **T5-Indonesian** untuk meringkas inti masalah.")

# 2. Load Models (Cached for speed)
@st.cache_resource
def load_sentiment_model():
    # Model khusus sentimen Indonesia yang lebih akurat (IndoBERTtweet)
    return pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")

@st.cache_resource
def load_summary_model():
    # Model khusus ringkasan teks Indonesia
    return pipeline("summarization", model="cahya/t5-base-indonesian-summarization-cased")

nlp_sentiment = load_sentiment_model()
nlp_summary = load_summary_model()

# 3. Input Section
st.subheader("📝 Masukkan Hingga 10 Komentar")
inputs = []
cols = st.columns(2)
for i in range(10):
    with cols[i % 2]:
        text = st.text_input(f"Komentar {i+1}:", key=f"input_{i}")
        if text.strip():
            inputs.append(text)

# 4. Analysis Logic
if st.button("Analisis & Ringkas"):
    if inputs:
        results_data = []
        combined_text = ". ".join(inputs) # Gabungkan semua teks untuk diringkas
        
        with st.spinner('AI sedang memahami konteks...'):
            # Part 1: Individual Sentiment
            for text in inputs:
                pred = nlp_sentiment(text)
                label = pred[0]['label'].lower()
                results_data.append({"Komentar": text, "Sentimen": label.capitalize()})
            
            # Part 2: Summarization (The context!)
            # Hanya ringkas jika teks cukup panjang
            if len(combined_text) > 50:
                summary = nlp_summary(combined_text, max_length=50, min_length=10, do_sample=False)
                context_summary = summary[0]['summary_text']
            else:
                context_summary = "Teks terlalu pendek untuk diringkas."

        # 5. Display
        df = pd.DataFrame(results_data)
        
        # Section A: Summary Box
        st.success("### ✨ Ringkasan Konteks (AI Insight)")
        st.write(f"**Inti dari komentar-komentar ini adalah:** *{context_summary}*")
        
        st.divider()
        
        # Section B: Charts and Tables
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📊 Statistik Sentimen")
            fig = px.pie(df, names='Sentimen', color='Sentimen', 
                         color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'})
            st.plotly_chart(fig)
            
        with col_right:
            st.subheader("📋 Detail Komentar")
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("Masukkan teks terlebih dahulu!")
        