import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Judul aplikasi
st.title("Prediksi Risiko Kredit Mobil Mazda")
st.write("Masukkan data debitur untuk memprediksi risiko kredit (0 = Lancar, 1 = Macet)")

# Daftar harga mobil Mazda (dalam Rupiah)
car_prices = {
    "Mazda 2": 371100000,
    "Mazda 3": 583800000,
    "Mazda CX-3": 403300000,
    "Mazda MX-5": 943300000,
    "Mazda CX-30": 585500000,
    "MAZDA CX-60": 1188800000,
    "MAZDA CX-5": 647700000,
    "MAZDA 6": 717700000,
    "MAZDA CX-9": 955500000,
    "MAZDA CX-8": 828800000,
    "MAZDA MX-30 EV": 860000000
}

# Form input data
st.subheader("Masukkan Data Debitur")
with st.form(key="credit_form"):
    car_name = st.selectbox("Pilih Nama Mobil", list(car_prices.keys()))
    income_total = st.number_input("Total Pendapatan (Rp)", min_value=0.0, value=15000000.0, step=1000000.0)
    dp_percent = st.number_input("Persen DP (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0)
    interest_rate = st.number_input("Suku Bunga per Tahun (%)", min_value=0.0, value=4.5, step=0.1) / (100 * 12)  # Konversi ke desimal
    tenor = st.number_input("Tenor (bulan)", min_value=1, value=60, step=1)
    age = st.number_input("Usia (tahun)", min_value=18, value=30, step=1)
    job = st.selectbox("Pekerjaan", ["Kelompok A", "Kelompok B", "Kelompok C", "Kelompok D", "Kelompok E"])
    blacklist = st.selectbox("Status Blacklist", ["Ada", "Tidak Ada"])
    debtor_type = st.selectbox("Jenis Debitur", ["Baru", "Lama"])
    dependents = st.number_input("Jumlah Tanggungan", min_value=0, value=2, step=1)
    house_status = st.selectbox("Status Rumah", ["Milik Sendiri", "Milik Orang Tua"])
    submit_button = st.form_submit_button(label="Prediksi")

# Proses data saat form disubmit
if submit_button:
    # Hitung variabel dinamis
    price_car = car_prices[car_name]
    dp_amount = price_car * (dp_percent / 100)
    loan_amount = price_car - dp_amount

    # Hitung cicilan bulanan berdasarkan metode reducing balance
    P = loan_amount
    r = interest_rate
    n = tenor
    if r > 0:
        monthly_payment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
    else:
        monthly_payment = P / n

    ratio_dti = (monthly_payment / income_total) * 100

    # Tampilkan perhitungan
    st.subheader("Hasil Perhitungan")
    st.write(f"**Harga Mobil**: Rp {price_car:,.2f}")
    st.write(f"**Jumlah DP**: Rp {dp_amount:,.2f}")
    st.write(f"**Jumlah Kredit**: Rp {loan_amount:,.2f}")
    st.write(f"**Cicilan Bulanan**: Rp {monthly_payment:,.2f}")
    st.write(f"**Rasio DTI**: {ratio_dti:.2f}%")

    # Buat dataframe untuk input model
    input_data = pd.DataFrame({
        'NAME_CAR': [car_name],
        'PRICE_CAR': [price_car],
        'PERCENT_DP': [dp_percent / 100],
        'INTEREST_RATE': [interest_rate],
        'DOWN_PAYMENT': [dp_amount],
        'AMT_CREDIT': [loan_amount],
        'TENOR': [tenor],
        'AMT_ANNUITY': [monthly_payment * 12],  # Annualized annuity
        'RASIO_DTI': [ratio_dti],
        'AGE': [age],
        'OCCUPATION_TYPE': [job],
        'DATA_BLACKLIST': [blacklist],
        'DEBITUR': [debtor_type],
        'TOTAL_DEPENDENTS': [dependents],
        'HOUSING_TYPE': [house_status]
    })

    # Load model dan scaler
    try:
        lr_model = joblib.load('lr_model.pkl')
        scaler = joblib.load('scaler.pkl')
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: File model, scaler, atau encoders tidak ditemukan. Pastikan sudah dilatih dan disimpan.")
        st.stop()

    # Preprocessing input data
    for col, le in encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col].astype(str))

    # Kolom untuk scaling
    scaled_cols = ['NAME_CAR', 'PRICE_CAR', 'PERCENT_DP', 'INTEREST_RATE', 'DOWN_PAYMENT',
                   'AMT_CREDIT', 'TENOR', 'AMT_ANNUITY', 'AGE',
                   'OCCUPATION_TYPE', 'DATA_BLACKLIST', 'DEBITUR', 'TOTAL_DEPENDENTS', 'HOUSING_TYPE']

    input_data_scaled = scaler.transform(input_data[scaled_cols])

    # Prediksi
    predik = lr_model.predict(input_data_scaled)[0]
    predik_prob = lr_model.predict_proba(input_data_scaled)[0][1]

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi Risiko**: {'Macet' if predik == 1 else 'Lancar'}")
    st.write(f"**Probabilitas Macet**: {predik_prob:.2f}")

    # Validasi sederhana
    if predik_prob > 0.5:
        st.warning("Peringatan: Risiko kredit cukup tinggi, pertimbangkan evaluasi lebih lanjut.")
    else:
        st.success("Kredit tampak aman berdasarkan data yang dimasukkan.")