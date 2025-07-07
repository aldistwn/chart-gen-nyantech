# main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="📊 FPS & CPU Chart Viewer", layout="wide")

st.title("🎮 Final Gaming Chart Viewer")
st.markdown("Upload CSV dari log performa dan tampilkan grafik **FPS** dan **CPU Usage** secara visual.")

# Upload
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        # Baca semua baris sebagai teks
        raw_lines = uploaded_file.readlines()
        cleaned_rows = []

        for line in raw_lines:
            decoded = line.decode("utf-8").strip()
            split_line = re.split(r'[\s,]+', decoded)
            row = [int(x) for x in split_line if x.isdigit()]
            if row:  # hanya baris valid
                cleaned_rows.append(row)

        df = pd.DataFrame(cleaned_rows)

        # Pastikan cukup kolom
        if df.shape[1] >= 5:
            fps = df[0]
            cpu = df[4]

            # Buat Chart
            st.subheader("📈 Line Chart")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(fps, label="FPS", color="skyblue", linewidth=1.5)
            ax.plot(cpu, label="CPU Usage (%)", color="orange", linewidth=1.5)
            ax.set_title("FPS & CPU Usage Over Time")
            ax.set_xlabel("Waktu (index)")
            ax.set_ylabel("Nilai")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("❗ Data tidak memiliki cukup kolom (minimal 5).")

    except Exception as e:
        st.error(f"❌ Gagal memproses file: {e}")
else:
    st.info("⬆️ Silakan upload file CSV untuk mulai.")
