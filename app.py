import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("Clustering Kabupaten/Kota di Sumatera Utara Berdasarkan Luas & Produksi Karet")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, skiprows=2)
    df.columns = [
        "Kabupaten_Kota",
        "Luas_2019", "Luas_2020", "Luas_2021",
        "Produksi_2019", "Produksi_2020", "Produksi_2021"
    ]
    df = df[df["Kabupaten_Kota"].notna()]
    df = df[~df["Kabupaten_Kota"].str.contains("Sumatera Utara", na=False)]

    # Ubah ke numerik dan bersihkan
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Hitung rata-rata
    df["Rata_Luas"] = df[["Luas_2019", "Luas_2020", "Luas_2021"]].mean(axis=1)
    df["Rata_Produksi"] = df[["Produksi_2019", "Produksi_2020", "Produksi_2021"]].mean(axis=1)

    # Normalisasi
    X = df[["Rata_Luas", "Rata_Produksi"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Tampilkan Elbow Curve
    st.subheader("Elbow Method")
    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, 10), inertia, marker='o')
    ax.set_title('Menentukan Jumlah Cluster Optimal (Elbow Method)')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    st.pyplot(fig_elbow)

    # Input jumlah cluster dari pengguna
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=6, value=3)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Tampilkan hasil clustering dalam scatterplot
    st.subheader("Hasil Clustering")
    fig_cluster, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Rata_Luas", y="Rata_Produksi",
        hue="Cluster", palette="Set1", s=100, ax=ax
    )
    for i in range(len(df)):
        x = df["Rata_Luas"][i]
        y = df["Rata_Produksi"][i]
        label = df["Kabupaten_Kota"][i]
        if pd.notnull(x) and pd.notnull(y) and x < 1e6 and y < 1e6:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 2), ha='left', fontsize=8)
    ax.set_title("Clustering Kabupaten/Kota Berdasarkan Rata-rata Luas & Produksi Karet")
    ax.set_xlabel("Rata-rata Luas Tanaman Karet (Ha)")
    ax.set_ylabel("Rata-rata Produksi Karet (Ton)")
    ax.legend(title="Cluster")
    ax.grid(True)
    st.pyplot(fig_cluster)

    # Tampilkan tabel hasil
    st.subheader("Tabel Hasil Clustering")
    st.dataframe(df[["Kabupaten_Kota", "Rata_Luas", "Rata_Produksi", "Cluster"]])

else:
    st.info("Silakan upload file CSV dengan format data produksi dan luas karet provinsi Sumut.")
