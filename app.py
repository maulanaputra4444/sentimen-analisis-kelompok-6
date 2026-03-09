import streamlit as st
from transformers import pipeline

# 1. Judul dan Deskripsi
st.set_page_config(page_title="Sentimen Analis X", page_icon="📊")
st.title("🔍 Alat Analisis Sentimen Indonesia")
st.write("Aplikasi ini menggunakan model AI (IndoRoBERTa) yang dilatih khusus untuk Bahasa Indonesia.")

# 2. Load Model (Disimpan di Cache agar cepat)
@st.cache_resource
def load_model():
    # Model ini dikembangkan oleh w11wo untuk sentimen Bahasa Indonesia
    return pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

nlp = load_model()

# 3. Area Input Pengguna
user_input = st.text_area("Tulis komentar di sini:", placeholder="Contoh: Makanannya enak banget tapi pelayanannya agak lama.")

# 4. Logika Analisis
if st.button("Analisis Sekarang"):
    if user_input.strip():
        with st.spinner('Menganalisis teks...'):
            # Melakukan prediksi
            result = nlp(user_input)
            label = result[0]['label'] # 'positive', 'neutral', atau 'negative'
            score = result[0]['score']
            
            # Pemetaan Label ke Format Anda
            mapping = {
                "positive": ("Positif 😊", "green"),
                "negative": ("Negatif 😡", "red"),
                "neutral": ("Netral 😐", "gray")
            }
            
            hasil, warna = mapping.get(label.lower(), ("Tidak Diketahui", "black"))
            
            # 5. Menampilkan Hasil
            st.divider()
            st.subheader("Hasil Analisis:")
            st.markdown(f"Komentar Anda terdeteksi: :{warna}[**{hasil}**]")
            st.info(f"Tingkat Keyakinan Model: {score:.2f}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")