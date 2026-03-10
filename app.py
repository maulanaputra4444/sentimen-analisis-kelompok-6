import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Sentimen & Ringkasan AI", page_icon="🤖", layout="wide")
st.title("🤖 Analisis Sentimen & Ringkasan Konteks Indonesia")
st.write("Aplikasi ini menggunakan **IndoBERT** untuk sentimen dan **T5** untuk meringkas inti masalah.")

# 2. Load Model (Gunakan cache agar aplikasi cepat)
@st.cache_resource
def load_sentiment_model():
    # Model khusus sentimen Indonesia
    return pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")

@st.cache_resource
def load_summary_model():
    # Force framework='pt' (PyTorch) untuk memperbaiki error 'Unknown Task'
    return pipeline(
        "summarization", 
        model="cahya/t5-base-indonesian-summarization-cased",
        framework="pt" 
    )

# Inisialisasi model
try:
    nlp_sentiment = load_sentiment_model()
    nlp_summary = load_summary_model()
except Exception as e:
    st.error(f"Gagal memuat model AI: {e}")
    st.stop()

# 3. Input 10 Komentar
st.subheader("📝 Masukkan Komentar (Maks. 10)")
inputs = []
cols = st.columns(2)
for i in range(10):
    with cols[i % 2]:
        text = st.text_input(f"Komentar {i+1}:", key=f"input_{i}")
        if text.strip():
            inputs.append(text)

# 4. Logika Analisis
if st.button("Analisis & Ringkas Sekarang"):
    if inputs:
        results_data = []
        combined_text = ". ".join(inputs) 
        
        with st.spinner('AI sedang bekerja...'):
            # Analisis Sentimen per Komentar
            for text in inputs:
                pred = nlp_sentiment(text)
                label = pred[0]['label'].lower()
                results_data.append({"Komentar": text, "Sentimen": label.capitalize()})
            
            # Ringkasan Konteks (Summarization)
            if len(combined_text) > 60:
                summary = nlp_summary(combined_text, max_length=50, min_length=10, do_sample=False)
                context_summary = summary[0]['summary_text']
            else:
                context_summary = "Teks terlalu pendek untuk diringkas."

        # 5. Menampilkan Hasil
        df = pd.DataFrame(results_data)
        
        st.success("### ✨ Ringkasan Konteks (Inti Masalah)")
        st.write(f"**AI Insight:** {context_summary}")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📊 Statistik Sentimen")
            fig = px.pie(df, names='Sentimen', color='Sentimen', 
                         color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.subheader("📋 Detail Tabel")
            st.dataframe(df, use_container_width=True)
            
    else:
        st.warning("Silakan isi komentar terlebih dahulu!")