import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentasi Demografis Penduduk",
    page_icon="👥",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD MODEL DARI .pkl
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        log_model = pickle.load(f)
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('scaler_supervised.pkl', 'rb') as f:
        scaler_sup = pickle.load(f)
    return kmeans, log_model, nb_model, scaler_sup

@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df_clean = df.copy()
    df_clean.drop_duplicates(inplace=True)
    df_clean.replace('?', np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    return df_clean

kmeans, log_model, nb_model, scaler_sup = load_models()
df_raw = load_data()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────
st.sidebar.title("👥 Segmentasi Demografis")
st.sidebar.markdown("**Tugas Besar Data Mining**")
st.sidebar.markdown("Kelompok 9")
st.sidebar.divider()

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🔮 Prediksi Segmentasi", "📊 Visualisasi Proses"]
)

st.sidebar.divider()
st.sidebar.markdown("""
**Dataset:**  
Adult Census Income  
UCI Machine Learning Repository, 1994

**Model:**
- K-Means Clustering (K=2)
- Logistic Regression (96.53%)
- Naïve Bayes (79.53%)

**Fitur Input (7 variabel):**
- age, education.num
- capital.gain, capital.loss
- hours.per.week
- sex, marital status
""")

# Profil cluster (hardcoded dari hasil analisis)
CLUSTER_PROFILES = {
    0: {
        "nama": "Pekerja Mapan & Berpengalaman",
        "emoji": "💼",
        "color": "#1a6e3c",
        "bg": "#e8f5e9",
        "stats": {
            "Rata-rata Usia": "~47 tahun",
            "Avg Education Num": "10.75",
            "Avg Jam Kerja": "~44 jam/minggu",
            "Avg Capital Gain": "~$1.829",
            "Dominan Gender": "Pria (76.9%)",
            "Income >50K": "41%"
        }
    },
    1: {
        "nama": "Pekerja Muda & Berkembang",
        "emoji": "🌱",
        "color": "#0d47a1",
        "bg": "#e3f2fd",
        "stats": {
            "Rata-rata Usia": "~28 tahun",
            "Avg Education Num": "9.34",
            "Avg Jam Kerja": "~36 jam/minggu",
            "Avg Capital Gain": "~$174",
            "Dominan Gender": "Pria (56%)",
            "Income >50K": "4.8%"
        }
    }
}

# ══════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI SEGMENTASI
# ══════════════════════════════════════════════
if page == "🔮 Prediksi Segmentasi":
    st.title("🔮 Prediksi Segmentasi Demografis")
    st.markdown(
        "Masukkan profil demografis seseorang untuk mengetahui "
        "mereka termasuk kelompok mana."
    )
    st.divider()

    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("📋 Form Input Data")

        with st.form("prediction_form"):
            age = st.slider("Usia", 17, 90, 35,
                            help="Usia penduduk dalam tahun")

            education_num = st.slider(
                "Tingkat Pendidikan (education.num)", 1, 16, 10,
                help="1=Tidak Sekolah · 9=SMA · 13=S1 · 14=S2 · 16=S3"
            )

            hours_per_week = st.slider(
                "Jam Kerja per Minggu", 1, 99, 40
            )

            capital_gain = st.number_input(
                "Capital Gain ($)", min_value=0, max_value=99999, value=0,
                help="Keuntungan dari investasi/aset"
            )

            capital_loss = st.number_input(
                "Capital Loss ($)", min_value=0, max_value=4356, value=0,
                help="Kerugian dari investasi/aset"
            )

            sex = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])

            marital = st.selectbox(
                "Status Pernikahan",
                ["Sudah Menikah", "Belum/Tidak Menikah"]
            )

            submitted = st.form_submit_button(
                "🔍 Prediksi Segmen",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        st.subheader("📌 Hasil Prediksi")

        if submitted:
            # Encode input
            sex_male       = 1 if sex == "Pria" else 0
            marital_married = 1 if marital == "Sudah Menikah" else 0

            # Susun input sesuai urutan training
            input_df = pd.DataFrame([[
                age, education_num, capital_gain,
                capital_loss, hours_per_week,
                sex_male, marital_married
            ]], columns=[
                'age', 'education.num', 'capital.gain',
                'capital.loss', 'hours.per.week',
                'sex_Male', 'marital_married'
            ])

            # Prediksi Logistic Regression
            input_scaled  = scaler_sup.transform(input_df)
            pred_log      = log_model.predict(input_scaled)[0]
            prob_log      = log_model.predict_proba(input_scaled)[0]

            # Prediksi Naive Bayes
            pred_nb       = nb_model.predict(input_df)[0]
            prob_nb       = nb_model.predict_proba(input_df)[0]

            profile = CLUSTER_PROFILES[pred_log]

            # Hasil utama
            st.markdown(f"""
            <div style='background:{profile["bg"]}; padding:20px;
                        border-radius:12px;
                        border-left:5px solid {profile["color"]};
                        margin-bottom:16px'>
                <h2 style='color:{profile["color"]}; margin:0'>
                    {profile["emoji"]} Cluster {pred_log}
                </h2>
                <h4 style='color:{profile["color"]}; margin:4px 0 0'>
                    {profile["nama"]}
                </h4>
            </div>
            """, unsafe_allow_html=True)

            # Probabilitas
            st.markdown("**Probabilitas Prediksi (Logistic Regression):**")
            p0, p1 = st.columns(2)
            p0.metric("Cluster 0 💼", f"{prob_log[0]*100:.1f}%")
            p1.metric("Cluster 1 🌱", f"{prob_log[1]*100:.1f}%")

            # Gauge bar probabilitas
            fig_prob, ax_prob = plt.subplots(figsize=(6, 1.2))
            ax_prob.barh([""], [prob_log[0]], color="#1a6e3c", label="Cluster 0")
            ax_prob.barh([""], [prob_log[1]], left=[prob_log[0]],
                         color="#0d47a1", label="Cluster 1")
            ax_prob.axvline(0.5, color='white', linestyle='--', linewidth=1.5)
            ax_prob.set_xlim(0, 1)
            ax_prob.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax_prob.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax_prob.legend(loc='upper right', fontsize=8)
            ax_prob.set_title("Distribusi Probabilitas", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close()

            # Perbandingan model
            st.markdown("**Perbandingan Hasil Model:**")
            comp_df = pd.DataFrame({
                "Model": ["Logistic Regression", "Naïve Bayes"],
                "Prediksi": [f"Cluster {pred_log} {CLUSTER_PROFILES[pred_log]['emoji']}",
                             f"Cluster {pred_nb} {CLUSTER_PROFILES[pred_nb]['emoji']}"],
                "Confidence": [f"{max(prob_log)*100:.1f}%",
                               f"{max(prob_nb)*100:.1f}%"],
                "Akurasi Model": ["96.53%", "79.53%"]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Karakteristik cluster
            st.markdown(f"**Karakteristik {profile['emoji']} Cluster {pred_log} — {profile['nama']}:**")
            for k, v in profile["stats"].items():
                st.markdown(f"- **{k}:** {v}")

            if pred_log != pred_nb:
                st.warning(
                    "⚠️ Kedua model memberikan prediksi berbeda. "
                    "Hasil Logistic Regression lebih direkomendasikan "
                    "karena akurasinya lebih tinggi (96.53%)."
                )
            else:
                st.success("✅ Kedua model sepakat pada prediksi yang sama!")

        else:
            st.info("👈 Isi form di sebelah kiri, lalu klik **Prediksi Segmen**")
            st.markdown("**Ringkasan Profil Cluster:**")
            st.markdown("""
| | Cluster 0 💼 | Cluster 1 🌱 |
|---|---|---|
| Profil | Pekerja Mapan | Pekerja Muda |
| Usia | ~47 tahun | ~28 tahun |
| Capital Gain | ~$1.829 | ~$174 |
| Jam Kerja | ~44 jam/minggu | ~36 jam/minggu |
| Income >50K | 41% | 4.8% |
| Gender Dominan | Pria 76.9% | Pria 56% |
            """)

# ══════════════════════════════════════════════
# HALAMAN 2 — VISUALISASI PROSES
# ══════════════════════════════════════════════
elif page == "📊 Visualisasi Proses":
    st.title("📊 Visualisasi Proses Data Mining")
    st.markdown(
        "Proses lengkap dari data preparation hingga evaluasi model "
        "Kelompok 9 — Analisis Segmentasi Demografis Penduduk."
    )

    # ── SECTION 1: DATA OVERVIEW ──────────────
    st.header("1️⃣ Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Record Awal", f"{len(df_raw):,}")
    c2.metric("Total Variabel", "15")
    c3.metric("Variabel Numerik", "6")
    c4.metric("Variabel Kategorikal", "9")

    with st.expander("👁️ Lihat Sampel Dataset (5 baris pertama)"):
        st.dataframe(df_raw.head(), use_container_width=True)

    with st.expander("📋 Keterangan Variabel"):
        var_df = pd.DataFrame({
            "Variabel": ["age", "workclass", "fnlwgt*", "education*",
                         "education.num", "marital.status", "occupation",
                         "relationship", "race", "sex", "capital.gain",
                         "capital.loss", "hours.per.week", "native.country",
                         "income**"],
            "Tipe": ["Numerik","Kategorikal","Numerik","Kategorikal",
                     "Numerik","Kategorikal","Kategorikal","Kategorikal",
                     "Kategorikal","Kategorikal","Numerik","Numerik",
                     "Numerik","Kategorikal","Kategorikal"],
            "Keterangan": [
                "Usia penduduk",
                "Sektor pekerjaan",
                "Bobot sensus (di-drop)",
                "Jenjang pendidikan (di-drop, sudah ada education.num)",
                "Tingkat pendidikan dalam angka (1-16)",
                "Status pernikahan",
                "Jenis pekerjaan",
                "Hubungan keluarga",
                "Ras/etnis",
                "Jenis kelamin",
                "Keuntungan modal/investasi",
                "Kerugian modal/investasi",
                "Jam kerja per minggu",
                "Negara asal",
                "Label income (di-exclude untuk clustering)"
            ]
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    # ── SECTION 2: DISTRIBUSI DATA ────────────
    st.header("2️⃣ Distribusi & Outlier Data")

    fig_dist, axes = plt.subplots(2, 3, figsize=(15, 8))

    plot_configs = [
        ('age', 'Distribusi Usia', 'steelblue'),
        ('education.num', 'Distribusi Tingkat Pendidikan', 'coral'),
        ('hours.per.week', 'Distribusi Jam Kerja/Minggu', 'mediumseagreen'),
        ('capital.gain', 'Distribusi Capital Gain', 'mediumpurple'),
        ('capital.loss', 'Distribusi Capital Loss', 'tomato'),
    ]

    for idx, (col, title, color) in enumerate(plot_configs):
        ax = axes[idx // 3][idx % 3]
        ax.hist(df_raw[col], bins=30, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel('Frekuensi')

    # Boxplot outlier di slot terakhir
    num_cols = ['age', 'education.num', 'hours.per.week',
                'capital.gain', 'capital.loss']
    axes[1][2].boxplot(
        [df_raw[c].dropna() for c in num_cols],
        labels=['age', 'edu', 'hours', 'c.gain', 'c.loss'],
        patch_artist=True
    )
    axes[1][2].set_title('Boxplot — Deteksi Outlier', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close()

    # Statistik outlier
    with st.expander("📊 Statistik Outlier per Variabel"):
        outlier_rows = []
        for col in num_cols:
            Q1 = df_raw[col].quantile(0.25)
            Q3 = df_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outlier = ((df_raw[col] < lower) | (df_raw[col] > upper)).sum()
            outlier_rows.append({
                "Variabel": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Batas Bawah": round(lower, 2),
                "Batas Atas": round(upper, 2),
                "Jumlah Outlier": n_outlier,
                "Persentase": f"{n_outlier/len(df_raw)*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(outlier_rows),
                     use_container_width=True, hide_index=True)
        st.info("ℹ️ Outlier tidak dihapus karena merepresentasikan kondisi nyata dalam data sensus.")

    # ── SECTION 3: CLUSTERING ─────────────────
    st.header("3️⃣ Proses Clustering — K-Means (K=2)")

    st.subheader("Penentuan K Optimal")
    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("K Optimal", "K = 2")
    col_k2.metric("Silhouette Score Tertinggi", "~0.75 (di K=2)")
    col_k3.metric("Metode", "Elbow + Silhouette")

    st.markdown("""
    **Justifikasi K=2:**
    Berdasarkan Silhouette Analysis, score tertinggi berada di K=2 (~0.75),
    yang menunjukkan pemisahan cluster paling optimal terjadi saat data
    dibagi menjadi 2 kelompok.
    """)

    # Profiling cluster
    st.subheader("Profiling Hasil Cluster")
    col_c0, col_c1 = st.columns(2)

    with col_c0:
        st.markdown("""
        <div style='background:#e8f5e9; padding:16px; border-radius:10px;
                    border-left:4px solid #1a6e3c'>
            <h4 style='color:#1a6e3c; margin:0'>💼 Cluster 0 — Pekerja Mapan</h4>
        </div>
        """, unsafe_allow_html=True)
        profile_0 = pd.DataFrame(CLUSTER_PROFILES[0]["stats"].items(),
                                  columns=["Atribut", "Nilai"])
        st.dataframe(profile_0, use_container_width=True, hide_index=True)

    with col_c1:
        st.markdown("""
        <div style='background:#e3f2fd; padding:16px; border-radius:10px;
                    border-left:4px solid #0d47a1'>
            <h4 style='color:#0d47a1; margin:0'>🌱 Cluster 1 — Pekerja Muda</h4>
        </div>
        """, unsafe_allow_html=True)
        profile_1 = pd.DataFrame(CLUSTER_PROFILES[1]["stats"].items(),
                                  columns=["Atribut", "Nilai"])
        st.dataframe(profile_1, use_container_width=True, hide_index=True)

    st.subheader("Kesenjangan Demografis Antar Cluster")
    kesenjangan_df = pd.DataFrame({
        "Aspek Kesenjangan": [
            "Usia", "Capital Gain", "Income >50K",
            "Proporsi Wanita", "Jam Kerja"
        ],
        "Cluster 0 💼": ["~47 tahun", "~$1.829", "41%", "23.1%", "~44 jam"],
        "Cluster 1 🌱": ["~28 tahun", "~$174", "4.8%", "44%", "~36 jam"],
        "Gap": ["~19 tahun", "~10x lipat", "~8.5x lipat", "+20.9%", "~8 jam"]
    })
    st.dataframe(kesenjangan_df, use_container_width=True, hide_index=True)

    # ── SECTION 4: EVALUASI MODEL ─────────────
    st.header("4️⃣ Evaluasi Model Supervised Learning")

    st.markdown("""
    **Fitur yang digunakan (7 variabel):**
    `age` · `education.num` · `capital.gain` · `capital.loss` ·
    `hours.per.week` · `sex_Male` · `marital_married`

    **Split data:** 80% Training · 20% Testing
    """)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Logistic Regression", "96.53%",
                  delta="Model Utama", delta_color="normal")
    col_m2.metric("Naïve Bayes", "79.53%",
                  delta="-17.00% vs LR", delta_color="inverse")

    # Bar chart perbandingan
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    models_name = ['Logistic\nRegression', 'Naïve\nBayes']
    scores = [0.9653, 0.7953]
    colors = ['steelblue', 'mediumseagreen']
    bars = ax_acc.bar(models_name, scores, color=colors,
                      edgecolor='white', width=0.4)
    ax_acc.set_ylim(0, 1.15)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Perbandingan Accuracy Model')
    for bar, score in zip(bars, scores):
        ax_acc.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{score*100:.2f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_acc)
    plt.close()

    # Confusion matrix hardcoded dari hasil training
    st.subheader("Confusion Matrix")
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))

    # Hardcoded dari hasil notebook
    cm_log = np.array([[3041, 258], [270, 2459]])
    cm_nb  = np.array([[1952, 1347], [99, 2630]])

    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0],
                xticklabels=['Cluster 0', 'Cluster 1'],
                yticklabels=['Cluster 0', 'Cluster 1'])
    axes_cm[0].set_title('Confusion Matrix\nLogistic Regression (96.53%)')
    axes_cm[0].set_xlabel('Prediksi')
    axes_cm[0].set_ylabel('Aktual')

    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes_cm[1],
                xticklabels=['Cluster 0', 'Cluster 1'],
                yticklabels=['Cluster 0', 'Cluster 1'])
    axes_cm[1].set_title('Confusion Matrix\nNaïve Bayes (79.53%)')
    axes_cm[1].set_xlabel('Prediksi')
    axes_cm[1].set_ylabel('Aktual')

    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

    with st.expander("📄 Interpretasi Confusion Matrix"):
        st.markdown("""
        **Logistic Regression:**
        - Cluster 0: 3.041 benar diprediksi, 258 salah → precision 92%
        - Cluster 1: 2.459 benar diprediksi, 270 salah → precision 91%
        - Model sangat konsisten di kedua cluster

        **Naïve Bayes:**
        - Cluster 0: precision tinggi (95%) tapi recall rendah (59%) →
          banyak anggota Cluster 0 yang salah diprediksi sebagai Cluster 1
        - Cluster 1: recall tinggi (96%) → hampir semua Cluster 1 terdeteksi
        - Kelemahan: asumsi independensi fitur tidak sepenuhnya terpenuhi
          pada data demografis
        """)

    # ── FOOTER ───────────────────────────────
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:gray; font-size:13px; padding:10px'>
        <b>Tugas Besar Data Mining — Kelompok 9</b><br>
        Analisis Segmentasi Demografis Penduduk Berdasarkan
        Karakteristik Sosial-Ekonomi Menggunakan Metode Clustering<br>
        Dataset: Adult Census Income · UCI Machine Learning Repository, 1994
    </div>
    """, unsafe_allow_html=True)
