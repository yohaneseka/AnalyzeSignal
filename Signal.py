import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

def linear_regression(x, y):
    """
    Fungsi untuk melakukan regresi linier.
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Menghitung koefisien regresi
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    # Menghitung nilai prediksi
    y_pred = a * x + b
    return y_pred, a, b

def dft(x):
    """
    Fungsi untuk melakukan Discrete Fourier Transform (DFT).
    """
    n = len(x)
    XrealDFT = np.zeros(n)
    XimajDFT = np.zeros(n)
    for k in range(n):
        for i in range(n):
            XrealDFT[k] += x[i] * np.cos(2 * np.pi * k * i / n)
            XimajDFT[k] -= x[i] * np.sin(2 * np.pi * k * i / n)
    
    # Magnitudo spektrum
    magDFT = np.sqrt(XrealDFT**2 + XimajDFT**2)
    return magDFT

def low_pass_filter(data, M, Fc, fs):
    """
    Fungsi untuk menerapkan filter low-pass menggunakan respons impuls.
    """
    m = M
    h = np.zeros(2 * m + 1)
    for j in range(-m, m + 1):
        if j == 0:
            h[j + m] = 2 * Fc / fs
        else:
            h[j + m] = np.sin(2 * np.pi * Fc * j / fs) / (np.pi * j)
    
    # Normalisasi koefisien filter
    h /= np.sum(h)

    # Aplikasi filter pada data
    filtered_data = np.convolve(data, h, mode='same')
    return filtered_data, h

# Judul Aplikasi
st.title("Analisis Bentuk Sinyal dengan Regresi Linier, DFT, dan LPF")
st.write("Unggah file .txt berisi data sinyal untuk melihat analisis bentuk sinyal, melakukan regresi linier, analisis DFT, dan menerapkan Low-Pass Filter (LPF).")

# Input File TXT
st.subheader("Unggah File TXT")
uploaded_file = st.file_uploader("Unggah file .txt dengan data sinyal (waktu dan amplitudo dipisahkan spasi):", type="txt")

if uploaded_file is not None:
    try:
        # Membaca file txt
        raw_data = uploaded_file.read().decode("utf-8")
        data_lines = raw_data.strip().split("\n")
        waktu, amplitudo = [], []

        for line in data_lines:
            w, a = map(float, line.split())
            waktu.append(w)
            amplitudo.append(a)

        waktu = np.array(waktu)
        amplitudo = np.array(amplitudo)

        st.write("Data Sinyal yang Diupload:")
        st.write("Waktu:", waktu)
        st.write("Amplitudo:", amplitudo)

        # Plot sinyal mentah
        st.subheader("Grafik Sinyal Mentah")
        fig, ax = plt.subplots()
        ax.plot(waktu, amplitudo, label="Sinyal Mentah")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Amplitudo")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Melakukan regresi linier
        st.subheader("Regresi Linier")
        y_pred, a, b = linear_regression(waktu, amplitudo)
        st.write(f"Persamaan regresi: y = {a:.4f}x + {b:.4f}")

        # Plot hasil regresi
        st.subheader("Grafik Hasil Regresi Linier")
        fig, ax = plt.subplots()
        ax.plot(waktu, amplitudo, label="Sinyal Mentah")
        ax.plot(waktu, y_pred, label="Regresi Linier", color="red")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Amplitudo")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Melakukan DFT
        st.subheader("Spektrum Frekuensi (DFT)")
        magDFT = dft(amplitudo)
        frekuensi = np.fft.fftfreq(len(amplitudo), d=(waktu[1] - waktu[0]))

        # Plot hasil DFT
        fig, ax = plt.subplots()
        ax.plot(frekuensi[:len(frekuensi)//2], magDFT[:len(magDFT)//2], label="Spektrum Magnitudo")
        ax.set_xlabel("Frekuensi")
        ax.set_ylabel("Magnitudo")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Menerapkan Low-Pass Filter
        st.subheader("Low-Pass Filter (LPF)")
        M = st.number_input("Masukkan jumlah tap (M):", min_value=1, value=10, step=1)
        Fc = st.number_input("Masukkan frekuensi cutoff (Fc):", min_value=1.0, value=50.0, step=1.0)
        fs = 125
        st.write(f"Frekuensi sampling (Fs): 125 Hz")

        filtered_amplitudo, impulse_response = low_pass_filter(amplitudo, M, Fc, fs)

        # Plot hasil filter
        st.subheader("Grafik Hasil Low-Pass Filter")
        fig, ax = plt.subplots()
        ax.plot(waktu, amplitudo, label="Sinyal Mentah")
        ax.plot(waktu, filtered_amplitudo, label="Sinyal Terfilter", color="green")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Amplitudo")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Plot respons impuls
        st.subheader("Respons Impuls Filter")
        fig, ax = plt.subplots()
        taps = np.arange(-M, M + 1)
        ax.plot(taps, impulse_response, marker="o", label="Respons Impuls")  # Menghubungkan titik-titik
        ax.set_xlabel("Index")
        ax.set_ylabel("Amplitudo")
        ax.set_title("Respons Impuls LPF")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah file .txt untuk memulai analisis.")
