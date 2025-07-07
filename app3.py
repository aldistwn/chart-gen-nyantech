# main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🎮 Final Gaming Chart Viewer", layout="wide")

st.title("📊 Gaming Chart Viewer (FPS & CPU)")
st.markdown("Upload CSV performa, dan aplikasi akan menampilkan grafik **FPS & CPU Usage** dengan waktu (menit:detik).")

uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        # Baca CSV langsung (karena format sudah rapi)
        df = pd.read_csv(uploaded_file)

        # Validasi kolom wajib
        if "FPS" in df.columns and "CPU(%)" in df.columns:
            fps = df["FPS"]
            cpu = df["CPU(%)"]

            # Buat kolom tambahan: Frame dan Time
            processed = pd.DataFrame({
                "Frame": np.arange(len(fps)),
                "Time": [f"{int(t//60)}:{int(t%60):02d}" for t in range(len(fps))],
                "FPS": fps,
                "CPUUsage": cpu
            })

            # Hitung limit FPS otomatis (150% dari average)
            fps_mean = fps.mean()
            fps_max = int(fps_mean * 1.5)

            # Buat dual axis chart
            st.subheader("📈 Grafik FPS & CPU Usage")
            fig, ax1 = plt.subplots(figsize=(14, 5))

            ax1.set_xlabel("Waktu (menit:detik)")
            ax1.set_ylabel("FPS", color="skyblue")
            ax1.plot(processed["Time"], processed["FPS"], color="skyblue", linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor="skyblue")
            ax1.set_ylim(0, fps_max)

            ax2 = ax1.twinx()
            ax2.set_ylabel("CPU Usage (%)", color="orange")
            ax2.plot(processed["Time"], processed["CPUUsage"], color="orange", linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor="orange")
            ax2.set_ylim(0, 100)

            plt.title("FPS & CPU Usage Over Time (Dual Axis)")
            ax1.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("❗ File tidak memiliki kolom 'FPS' dan 'CPU(%)'. Periksa header CSV Anda.")

    except Exception as e:
        st.error(f"❌ Gagal memproses file: {e}")
else:
    st.info("⬆️ Silakan upload file CSV untuk mulai.")
