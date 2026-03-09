import streamlit as st
from textblob import TextBlob
from deep_translator import GoogleTranslator

# 1. Judul dan Deskripsi
st.set_page_config(page_title="Sentimen Analis X", page_icon="📊")
st.title("🔍 Alat Analisis Sentimen Sederhana")
st.write("Masukkan komentar dalam Bahasa Indonesia, dan kami akan menganalisisnya.")

# 2. Area Input Pengguna
user_input = st.text_area("Tulis komentar di sini:", placeholder="Contoh: Saya sangat senang dengan fitur baru ini!")

# 3. Logika Analisis
if st.button("Analisis Sekarang"):
    if user_input.strip():
        try:
            # Langkah Penting: Terjemahkan ke Inggris karena TextBlob butuh Bahasa Inggris
            translated = GoogleTranslator(source='auto', target='en').translate(user_input)
            
            # Proses teks hasil terjemahan menggunakan TextBlob
            blob = TextBlob(translated)
            skor = blob.sentiment.polarity # Skala -1 sampai 1
            
            # Penentuan Label
            if skor > 0:
                hasil = "Positif 😊"
                warna = "green"
            elif skor < 0:
                hasil = "Negatif 😡"
                warna = "red"
            else:
                hasil = "Netral 😐"
                warna = "gray"
            
            # 4. Menampilkan Hasil
            st.divider()
            st.subheader("Hasil Analisis:")
            st.markdown(f"Komentar Anda terdeteksi: :{warna}[**{hasil}**]")
            st.info(f"Skor Polaritas: {skor:.2f}")
            st.caption(f"Terjemahan (untuk sistem): {translated}")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menerjemahkan: {e}")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")