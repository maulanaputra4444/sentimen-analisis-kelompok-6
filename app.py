import streamlit as st
from textblob import TextBlob

# 1. Judul dan Deskripsi
st.set_page_config(page_title="Sentimen Analis X", page_icon="📊")
st.title("🔍 Alat Analisis Sentimen Sederhana")
st.write("Masukkan komentar atau tweet di bawah ini untuk melihat apakah nadanya Positif, Negatif, atau Netral.")

# 2. Area Input Pengguna
user_input = st.text_area("Tulis komentar di sini:", placeholder="Contoh: Saya sangat senang dengan fitur baru ini!")

# 3. Logika Analisis
if st.button("Analisis Sekarang"):
    if user_input:
        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        # Proses teks menggunakan TextBlob
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
        st.info(f"Skor Polaritas: {skor:.2f} (Skala -1 hingga 1)")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")

