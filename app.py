import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Sentimen Analis X - Batch", page_icon="📊", layout="wide")
st.title("🔍 Analisis Sentimen Komentar Media Sosial")
st.write("Masukkan hingga 10 komentar di bawah ini untuk melihat statistik sentimen kelompok Anda.")

# 2. Load Model (Cached)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

nlp = load_model()

# 3. Input Section - Using a loop for 10 inputs
st.subheader("📝 Masukkan Komentar")
inputs = []
cols = st.columns(2) # Splitting into 2 columns to save space

for i in range(10):
    with cols[i % 2]: # Alternates between column 0 and 1
        text = st.text_input(f"Komentar {i+1}:", key=f"input_{i}", placeholder=f"Ketik komentar ke-{i+1}...")
        if text.strip():
            inputs.append(text)

# 4. Analysis Logic
if st.button("Analisis Semua Komentar"):
    if inputs:
        results_data = []
        
        with st.spinner(f'Menganalisis {len(inputs)} komentar...'):
            for text in inputs:
                prediction = nlp(text)
                label = prediction[0]['label'].lower()
                score = prediction[0]['score']
                results_data.append({"Komentar": text, "Sentimen": label.capitalize(), "Skor": score})
        
        # Convert to DataFrame for easy handling
        df = pd.DataFrame(results_data)

        # 5. Display Results & Visualization
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("📋 Tabel Hasil")
            st.dataframe(df, use_container_width=True)

        with res_col2:
            st.subheader("📈 Distribusi Sentimen")
            # Create Pie Chart
            fig = px.pie(
                df, 
                names='Sentimen', 
                color='Sentimen',
                color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'},
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("Silakan isi setidaknya satu komentar!")