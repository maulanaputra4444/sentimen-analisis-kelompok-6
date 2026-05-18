import io
import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
from fpdf import FPDF

# ── 1. Konfigurasi halaman ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimen Analis X - Batch",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Analisis Sentimen Komentar Media Sosial")
st.write("Masukkan hingga 10 komentar di bawah ini")

# ── 2. Load Model (di-cache agar tidak muat ulang) ─────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier"
    )

nlp = load_model()

# ── 3. Helper: konversi DataFrame → Excel ──────────────────────────────────
def to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Analisis")

        # Styling kolom agar lebar otomatis
        ws = writer.sheets["Hasil Analisis"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    return buffer.getvalue()

# ── 4. Helper: konversi DataFrame → PDF ────────────────────────────────────
def to_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Judul
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Hasil Analisis Sentimen Komentar", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total komentar dianalisis: {len(df)}", ln=True, align="C")
    pdf.ln(6)

    # Ringkasan per label
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Ringkasan", ln=True)
    pdf.set_font("Helvetica", "", 10)
    summary = df["Sentimen"].value_counts()
    for label, count in summary.items():
        pct = count / len(df) * 100
        pdf.cell(0, 7, f"  {label}: {count} komentar ({pct:.1f}%)", ln=True)
    pdf.ln(4)

    # Header tabel
    col_widths = [100, 35, 35]
    headers = ["Komentar", "Sentimen", "Skor"]
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 9, h, border=1, fill=True)
    pdf.ln()

    # Isi tabel
    pdf.set_font("Helvetica", "", 9)
    for _, row in df.iterrows():
        komentar = str(row["Komentar"])
        if len(komentar) > 55:
            komentar = komentar[:55] + "..."
        sentimen = str(row["Sentimen"])
        skor = f"{row['Skor']:.2%}"
        pdf.cell(col_widths[0], 8, komentar, border=1)
        pdf.cell(col_widths[1], 8, sentimen, border=1, align="C")
        pdf.cell(col_widths[2], 8, skor, border=1, align="C")
        pdf.ln()

    return bytes(pdf.output())

# ── 5. Input komentar ───────────────────────────────────────────────────────
st.subheader("📝 Masukkan Komentar")
inputs = []
cols = st.columns(2)

for i in range(10):
    with cols[i % 2]:
        text = st.text_input(
            f"Komentar {i + 1}:",
            key=f"input_{i}",
            placeholder=f"Ketik komentar ke-{i + 1}..."
        )
        if text.strip():
            inputs.append(text)

# ── 6. Tombol analisis ──────────────────────────────────────────────────────
if st.button("🔍 Analisis Semua Komentar"):
    if inputs:
        results_data = []
        with st.spinner(f"Menganalisis {len(inputs)} komentar..."):
            for text in inputs:
                prediction = nlp(text)
                label = prediction[0]["label"].lower()
                score = prediction[0]["score"]
                results_data.append({
                    "Komentar": text,
                    "Sentimen": label.capitalize(),
                    "Skor": score
                })

        df = pd.DataFrame(results_data)

        # ── 7. Tampilkan hasil + visualisasi ───────────────────────────────
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("📋 Tabel Hasil")
            # Tampilkan skor sebagai persen agar lebih mudah dibaca
            df_display = df.copy()
            df_display["Skor"] = df_display["Skor"].map(lambda x: f"{x:.2%}")
            st.dataframe(df_display, use_container_width=True)

        with res_col2:
            st.subheader("📈 Distribusi Sentimen")
            fig = px.pie(
                df,
                names="Sentimen",
                color="Sentimen",
                color_discrete_map={
                    "Positive": "#2ca02c",
                    "Negative": "#d62728",
                    "Neutral":  "#7f7f7f"
                },
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 8. Tombol Export ───────────────────────────────────────────────
        st.divider()
        st.subheader("💾 Ekspor Hasil")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            excel_bytes = to_excel(df)
            st.download_button(
                label="⬇️ Download Excel (.xlsx)",
                data=excel_bytes,
                file_name="hasil_sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with exp_col2:
            pdf_bytes = to_pdf(df)
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="hasil_sentimen.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    else:
        st.warning("Silakan isi setidaknya satu komentar!")
