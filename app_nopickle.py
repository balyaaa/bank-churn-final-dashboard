
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("🔍 Customer Churn Prediction Dashboard")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("Customer-Churn-Records.csv")
    X = df.drop(columns=["CustomerId", "Surname", "Exited"], errors="ignore")
    y = df["Exited"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    model.fit(X, y)
    return model, df

model, full_df = load_and_train_model()

uploaded_file = st.file_uploader("Upload customer data (.csv)", type=["csv"])

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)

    # Filter Sidebar
    st.sidebar.header("🔧 Filter Pelanggan")
    if "Geography" in df_new.columns:
        geo_filter = st.sidebar.multiselect("Pilih Negara", options=df_new["Geography"].unique(), default=df_new["Geography"].unique())
        df_new = df_new[df_new["Geography"].isin(geo_filter)]

    if "Gender" in df_new.columns:
        gender_filter = st.sidebar.multiselect("Pilih Gender", options=df_new["Gender"].unique(), default=df_new["Gender"].unique())
        df_new = df_new[df_new["Gender"].isin(gender_filter)]

    if "Age" in df_new.columns:
        age_min, age_max = int(df_new["Age"].min()), int(df_new["Age"].max())
        age_range = st.sidebar.slider("Rentang Usia", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        df_new = df_new[df_new["Age"].between(age_range[0], age_range[1])]

    st.write("### Preview Data yang Difilter")
    st.dataframe(df_new.head())

    # Predict
    pred_input = df_new.drop(columns=["CustomerId", "Surname"], errors='ignore')
    churn_probs = model.predict_proba(pred_input)[:, 1]
    predictions = model.predict(pred_input)
    df_new["Churn Prediction"] = predictions
    df_new["Churn Probability"] = churn_probs.round(3)

    churn_count = df_new["Churn Prediction"].value_counts()
    churn_percent = churn_count / len(df_new) * 100

    st.write("### Prediction Results")
    st.dataframe(df_new[["CustomerId", "Churn Prediction", "Churn Probability"]])

    st.write("### 📊 Churn Summary")
    st.metric("Churned Customers", churn_count.get(1, 0))
    st.metric("Churn Rate", f"{churn_percent.get(1, 0):.2f}%")

    st.write("### 📈 Visualisasi")
    pie_data = df_new["Churn Prediction"].value_counts().rename({0: "Not Churn", 1: "Churn"})
    st.plotly_chart(px.pie(values=pie_data.values, names=pie_data.index, title="Proporsi Churn"))

    if "Age" in df_new.columns:
        st.plotly_chart(px.histogram(df_new, x="Age", color="Churn Prediction", barmode="overlay", title="Distribusi Usia vs Churn"))

    st.write("### 👤 Analisis Individu")
    selected_id = st.selectbox("Pilih Customer ID", df_new["CustomerId"].unique())
    selected_row = df_new[df_new["CustomerId"] == selected_id].T
    st.dataframe(selected_row.rename(columns={selected_row.columns[0]: "Value"}))

    csv_download = df_new.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Hasil Prediksi (CSV)", data=csv_download, file_name="churn_predictions.csv", mime="text/csv")

else:
    st.info("Silakan upload file CSV yang berisi data pelanggan untuk diprediksi.")

# Footer Credit
st.markdown("---")
st.markdown("**Dibuat oleh: Iqbal Alfaridzi Hakim**")
st.markdown("📧 Email: alfaridzihakim@gmail.com")
st.markdown("📷 Instagram: [@i2baal](https://instagram.com/i2baal)")
st.markdown("💼 LinkedIn: [iqbal-alfaridzi-hakim](https://www.linkedin.com/in/iqbal-alfaridzi-hakim-2763a6254/)")
st.markdown("🐙 GitHub: [balyaaa](https://github.com/balyaaa)")
