import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Analisis Sentimen & Ringkasan AI", page_icon="🤖", layout="wide")
st.title("🤖 Analisis Sentimen & Ringkasan Konteks Indonesia")
st.write("Aplikasi ini menggunakan **IndoBERT** untuk sentimen dan **T5-Indonesian** untuk meringkas inti masalah.")

# 2. Load Models (Cached for speed)
@st.cache_resource
def load_sentiment_model():
    # Model khusus sentimen Indonesia (IndoBERTtweet)
    return pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")

@st.cache_resource
def load_summary_model():
    # Force 'pt' (PyTorch) to ensure the 'summarization' task is recognized
    return pipeline(
        "summarization", 
        model="cahya/t5-base-indonesian-summarization-cased",
        framework="pt"
    )

try:
    nlp_sentiment = load_sentiment_model()
    nlp_summary = load_summary_model()
except Exception as e:
    st.error(f"Gagal memuat model AI: {e}")
    st.stop()

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
        combined_text = ". ".join(inputs) 
        
        with st.spinner('AI sedang memahami konteks...'):
            # Part 1: Analisis Sentimen Individual
            for text in inputs:
                pred = nlp_sentiment(text)
                label = pred[0]['label'].lower()
                results_data.append({"Komentar": text, "Sentimen": label.capitalize()})
            
            # Part 2: Ringkasan (The context!)
            if len(combined_text) > 50:
                # T5 needs specific length constraints
                summary = nlp_summary(combined_text, max_length=50, min_length=10, do_sample=False)
                context_summary = summary[0]['summary_text']
            else:
                context_summary = "Teks terlalu pendek untuk diringkas otomatis."

        # 5. Display Results
        df = pd.DataFrame(results_data)
        
        st.success("### ✨ Ringkasan Konteks (AI Insight)")
        st.write(f"**Inti dari komentar-komentar ini adalah:** *{context_summary}*")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📊 Statistik Sentimen")
            fig = px.pie(df, names='Sentimen', color='Sentimen', 
                         color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.subheader("📋 Detail Komentar")
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("Masukkan setidaknya satu teks terlebih dahulu!")