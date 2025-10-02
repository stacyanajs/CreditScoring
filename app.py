import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Judul dan deskripsi
st.title("Prediksi Risiko Kredit Mobil Mazda dengan Logistic Regression")
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

# Cache model dan preprocessors untuk efisiensi
@st.cache_resource
def load_model_and_preprocessors():
    try:
        lr_model = joblib.load('lr_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        # Ensure NAME_CAR encoder includes all car names
        if 'NAME_CAR' in encoders:
            le = LabelEncoder()
            le.fit(list(car_prices.keys()))
            encoders['NAME_CAR'] = le
        return lr_model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"Error: File tidak ditemukan - {e}. Pastikan lr_model.pkl, scaler.pkl, dan encoders.pkl ada di direktori.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

# Load model dan preprocessors
lr_model, scaler, encoders = load_model_and_preprocessors()

if lr_model is None or scaler is None or encoders is None:
    st.stop()

# Form input
st.subheader("Masukkan Data Debitur")
with st.form(key="credit_form"):
    car_name = st.selectbox("Pilih Nama Mobil", list(car_prices.keys()))
    income_total = st.number_input("Total Pendapatan (Rp)", min_value=0.0, value=15000000.0, step=1000000.0)
    dp_percent = st.number_input("Persen DP (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0)
    interest_rate = st.number_input("Suku Bunga per Tahun (%)", min_value=0.0, value=4.5, step=0.1) / (100 * 12)
    tenor = st.number_input("Tenor (bulan)", min_value=1, value=60, step=1)
    age = st.number_input("Usia (tahun)", min_value=18, value=30, step=1)
    job = st.selectbox("Pekerjaan", ["Kelompok A", "Kelompok B", "Kelompok C", "Kelompok D", "Kelompok E"])
    blacklist = st.selectbox("Status Blacklist", ["Ada", "Tidak Ada"])
    debtor_type = st.selectbox("Jenis Debitur", ["Baru", "Lama"])
    dependents = st.number_input("Jumlah Tanggungan", min_value=0, value=2, step=1)
    house_status = st.selectbox("Status Rumah", ["Milik Sendiri", "Milik Orang Tua"])
    submit_button = st.form_submit_button(label="Prediksi")

if submit_button:
    # Hitung variabel dinamis
    price_car = car_prices[car_name]
    dp_amount = price_car * (dp_percent / 100)
    loan_amount = price_car - dp_amount
    monthly_payment = loan_amount * (interest_rate * (1 + interest_rate)**tenor) / ((1 + interest_rate)**tenor - 1) if interest_rate > 0 else loan_amount / tenor
    ratio_dti = (monthly_payment / income_total) * 100 if income_total > 0 else 0

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
        'AMT_ANNUITY': [monthly_payment * 12],  # Annualized
        'RASIO_DTI': [ratio_dti],
        'AGE': [age],
        'OCCUPATION_TYPE': [job],
        'DATA_BLACKLIST': [blacklist],
        'DEBITUR': [debtor_type],
        'TOTAL_DEPENDENTS': [dependents],
        'HOUSING_TYPE': [house_status]
    })

    # Encode categorical columns
    encoded_data = input_data.copy()
    for col, le in encoders.items():
        if col in input_data.columns:
            try:
                encoded_data[col] = le.transform(input_data[col].astype(str))
                st.write(f"Encoded {col}: {input_data[col].iloc[0]} -> {encoded_data[col].iloc[0]}")
            except ValueError as e:
                st.error(f"Error encoding {col}: {e}. Pastikan kategori valid sesuai pelatihan.")
                st.stop()

    # Define columns used for scaling (adjust based on training)
    scaled_cols = ['NAME_CAR', 'PRICE_CAR', 'PERCENT_DP', 'INTEREST_RATE', 'DOWN_PAYMENT',
                   'AMT_CREDIT', 'TENOR', 'AMT_ANNUITY', 'AGE',
                   'OCCUPATION_TYPE', 'DATA_BLACKLIST', 'DEBITUR', 'TOTAL_DEPENDENTS', 'HOUSING_TYPE']

    # Scale input data
    try:
        input_data_scaled = scaler.transform(encoded_data[scaled_cols])
        st.write("\n**Data setelah scaling**:")
        st.dataframe(pd.DataFrame(input_data_scaled, columns=scaled_cols))
    except ValueError as e:
        st.error(f"Error scaling data: {e}. Pastikan kolom sesuai dengan pelatihan.")
        st.stop()

    # Prediksi
    threshold = 0.4  # Adjusted to increase sensitivity to Macet
    lr_prob = lr_model.predict_proba(input_data_scaled)[0][1]  # Probabilitas Macet
    lr_pred = 1 if lr_prob > threshold else 0

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi Risiko**: {'Macet' if lr_pred == 1 else 'Lancar'}")
    st.write(f"**Probabilitas Macet**: {lr_prob:.2f}")
    st.write(f"**Threshold**: {threshold}")

    # Validasi sederhana
    if lr_pred == 1:
        st.warning("Peringatan: Risiko kredit cukup tinggi, pertimbangkan evaluasi lebih lanjut.")
    else:
        st.success("Kredit tampak aman berdasarkan data yang dimasukkan.")
