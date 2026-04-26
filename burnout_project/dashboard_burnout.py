# ============================================================
#  BURNOUT RISK PREDICTION SYSTEM – Streamlit Dashboard
#  Jalankan dengan: streamlit run dashboard_burnout.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Konfigurasi Halaman ─────────────────────────────────────
st.set_page_config(
    page_title="Burnout Risk Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Kustom ──────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}
.risk-high   { background-color: #ff4444; color: white; padding: 8px 16px;
               border-radius: 8px; font-weight: bold; }
.risk-medium { background-color: #ff8800; color: white; padding: 8px 16px;
               border-radius: 8px; font-weight: bold; }
.risk-low    { background-color: #00aa44; color: white; padding: 8px 16px;
               border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Load Artefak Model ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pkl_path = "model_artifacts.pkl"
    if not os.path.exists(pkl_path):
        # coba path outputs
        pkl_path = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    csv_path = "df_clean.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "df_clean.csv")
    return pd.read_csv(csv_path)

try:
    artifacts = load_artifacts()
    df = load_data()
    model        = artifacts["best_model"]
    scaler       = artifacts["scaler"]
    le_gender    = artifacts["le_gender"]
    le_role      = artifacts["le_role"]
    FEATURES     = artifacts["features"]
    model_name   = artifacts["best_model_name"]
    results_df   = pd.DataFrame(artifacts["results_df"]).T
    DATA_LOADED  = True
except Exception as e:
    DATA_LOADED = False
    st.error(f"❌ Gagal memuat model artifacts: {e}\n\n"
             "Pastikan file `model_artifacts.pkl` dan `df_clean.csv` "
             "berada di folder yang sama dengan `dashboard_burnout.py`.")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/fire.png", width=80)
st.sidebar.title("🔥 Burnout Risk Prediction")
st.sidebar.markdown(f"**Model aktif:** `{model_name}`")
st.sidebar.divider()

page = st.sidebar.selectbox(
    "📌 Navigasi Halaman",
    ["🏠 Overview Dataset",
     "📊 Exploratory Data Analysis",
     "🤖 Prediksi Individu",
     "📈 Performa Model",
     "🧪 A/B Testing"]
)

st.sidebar.divider()
st.sidebar.markdown("**Capstone Project – Data Science**")
st.sidebar.markdown("Burnout Risk Prediction System")

# ═══════════════════════════════════════════════════════════════
# HALAMAN 1: OVERVIEW DATASET
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview Dataset":
    st.title("🏠 Overview Dataset")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Karyawan", f"{len(df):,}", help="Jumlah total data")
    with col2:
        burnout_count = df["Burnout"].sum()
        st.metric("Karyawan Burnout", f"{int(burnout_count):,}",
                  delta=f"{burnout_count/len(df)*100:.1f}%", delta_color="inverse")
    with col3:
        avg_stress = df["StressLevel"].mean()
        st.metric("Rata-rata Stres", f"{avg_stress:.2f} / 10")
    with col4:
        avg_hours = df["WorkHoursPerWeek"].mean()
        st.metric("Rata-rata Jam Kerja", f"{avg_hours:.1f} jam/minggu")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📋 Statistik Deskriptif")
        num_cols = ["Age", "WorkHoursPerWeek", "StressLevel",
                    "SatisfactionLevel", "Experience"]
        st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    with col_b:
        st.subheader("🎯 Distribusi RiskLevel")
        risk_counts = df["RiskLevel"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ["#4CAF50", "#FF9800", "#F44336"]
        order = ["Rendah", "Sedang", "Tinggi"]
        ax.pie([risk_counts.get(r, 0) for r in order],
               labels=order, colors=colors, autopct="%1.1f%%",
               startangle=140, textprops={"fontsize": 12})
        ax.set_title("Proporsi Risiko Burnout")
        st.pyplot(fig)
        plt.close()

    st.subheader("📄 Sample Data (10 baris pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📖 Data Dictionary")
    dict_cols = {
        "Kolom": ["Age","Gender","JobRole","Experience","WorkHoursPerWeek",
                  "RemoteRatio","SatisfactionLevel","StressLevel","Burnout","RiskLevel"],
        "Tipe":  ["int","str","str","int","int","int","float","int","int (0/1)","str"],
        "Deskripsi": [
            "Usia karyawan (tahun)",
            "Jenis kelamin (Male/Female)",
            "Posisi jabatan (Analyst/Engineer/Manager/Sales/HR)",
            "Lama pengalaman kerja (tahun)",
            "Rata-rata jam kerja per minggu",
            "Persentase kerja remote (%)",
            "Skor kepuasan kerja (1–5)",
            "Tingkat stres (1–10)",
            "Label target: 0=Tidak Burnout, 1=Burnout",
            "Label risiko: Rendah / Sedang / Tinggi",
        ]
    }
    st.dataframe(pd.DataFrame(dict_cols), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# HALAMAN 2: EDA
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribusi Fitur", "Korelasi", "Burnout per Kategori", "Boxplot Perbandingan"]
    )

    with tab1:
        st.subheader("Distribusi Fitur Numerik")
        feat_sel = st.selectbox("Pilih fitur:", ["WorkHoursPerWeek","StressLevel",
                                                  "SatisfactionLevel","Age","Experience","RemoteRatio"])
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df.copy()
        df_plot["Status"] = df_plot["Burnout"].map({0: "Tidak Burnout", 1: "Burnout"})
        sns.histplot(data=df_plot, x=feat_sel, hue="Status", bins=25,
                     kde=True, ax=ax,
                     palette={"Tidak Burnout": "#2196F3", "Burnout": "#F44336"})
        ax.set_title(f"Distribusi {feat_sel} berdasarkan Status Burnout")
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Burnout = 0 (Tidak Burnout)**")
            st.dataframe(df[df["Burnout"]==0][feat_sel].describe().round(2).to_frame())
        with col2:
            st.markdown("**Burnout = 1 (Burnout)**")
            st.dataframe(df[df["Burnout"]==1][feat_sel].describe().round(2).to_frame())

    with tab2:
        st.subheader("Heatmap Korelasi Antar Fitur")
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
        ax.set_title("Korelasi Antar Fitur Numerik")
        st.pyplot(fig)
        plt.close()

        st.subheader("📊 Korelasi terhadap Burnout (diurutkan)")
        corr_burnout = corr["Burnout"].drop("Burnout").sort_values(key=abs, ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_burnout.values]
        ax2.barh(corr_burnout.index, corr_burnout.values, color=colors)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_title("Korelasi Fitur terhadap Burnout")
        ax2.set_xlabel("Pearson Correlation")
        st.pyplot(fig2)
        plt.close()

    with tab3:
        st.subheader("Burnout Rate per Kategori")
        cat_col = st.selectbox("Pilih kategori:", ["JobRole", "Gender", "RiskLevel"])
        burnout_rate = df.groupby(cat_col)["Burnout"].mean() * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        palette_use = sns.color_palette("husl", len(burnout_rate))
        bars = ax.bar(burnout_rate.index, burnout_rate.values,
                      color=palette_use, edgecolor="black")
        ax.set_ylabel("Burnout Rate (%)")
        ax.set_title(f"Burnout Rate per {cat_col}")
        for bar, val in zip(bars, burnout_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f"{val:.1f}%", ha="center", fontsize=11)
        st.pyplot(fig)
        plt.close()
        st.dataframe(burnout_rate.reset_index().rename(
            columns={"Burnout": "Burnout Rate (%)"}
        ).round(2), use_container_width=True)

    with tab4:
        st.subheader("Boxplot: Burnout vs Tidak Burnout")
        box_feat = st.selectbox("Pilih fitur:", ["WorkHoursPerWeek", "StressLevel",
                                                  "SatisfactionLevel", "Age"])
        df_box = df.copy()
        df_box["Status"] = df_box["Burnout"].map({0: "Tidak Burnout", 1: "Burnout"})
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_box, x="Status", y=box_feat,
                    palette={"Tidak Burnout": "#2196F3", "Burnout": "#F44336"}, ax=ax)
        ax.set_title(f"Perbandingan {box_feat}: Burnout vs Tidak Burnout")
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            mean_nb = df[df["Burnout"]==0][box_feat].mean()
            st.info(f"Rata-rata {box_feat} (Tidak Burnout): **{mean_nb:.2f}**")
        with col2:
            mean_b = df[df["Burnout"]==1][box_feat].mean()
            st.error(f"Rata-rata {box_feat} (Burnout): **{mean_b:.2f}**")


# ═══════════════════════════════════════════════════════════════
# HALAMAN 3: PREDIKSI INDIVIDU
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Prediksi Individu":
    st.title("🤖 Prediksi Risiko Burnout Individu")
    st.markdown("Masukkan data karyawan untuk mendapatkan prediksi risiko burnout.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age         = st.slider("🎂 Usia (tahun)", 22, 60, 35)
        experience  = st.slider("💼 Pengalaman Kerja (tahun)", 0, 39, 5)
        work_hours  = st.slider("⏰ Jam Kerja per Minggu", 30, 70, 50)
        remote      = st.slider("🏠 Remote Ratio (%)", 0, 100, 50)
    with col2:
        satisfaction = st.slider("😊 Kepuasan Kerja (1–5)", 1.0, 5.0, 3.0, 0.1)
        stress       = st.slider("😰 Tingkat Stres (1–10)", 1, 10, 5)
        gender       = st.selectbox("👤 Gender", ["Male", "Female"])
        job_role     = st.selectbox("💼 Posisi / Job Role",
                                    ["Analyst", "Engineer", "HR", "Manager", "Sales"])

    st.markdown("---")

    if st.button("🔍 Prediksi Risiko Burnout", type="primary", use_container_width=True):
        # Encode input
        gender_enc   = le_gender.transform([gender])[0]
        role_enc     = le_role.transform([job_role])[0]
        stress_ratio = stress / (work_hours + 1)
        wl_score     = satisfaction * (1 - remote / 100)
        high_risk    = int(stress >= 7 and work_hours >= 50)
        senior       = int(experience >= 10)

        input_data = pd.DataFrame([[
            age, experience, work_hours, remote, satisfaction, stress,
            gender_enc, role_enc, stress_ratio, wl_score, high_risk, senior
        ]], columns=FEATURES)

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        burnout_prob = proba[1] * 100

        # Tentukan risk level manual
        score = 0
        if work_hours >= 55: score += 2
        elif work_hours >= 45: score += 1
        if stress >= 7: score += 2
        elif stress >= 4: score += 1
        if satisfaction <= 2.0: score += 2
        elif satisfaction <= 3.5: score += 1
        if score >= 4: risk_lvl = "Tinggi"
        elif score >= 2: risk_lvl = "Sedang"
        else: risk_lvl = "Rendah"

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **BURNOUT TERDETEKSI**\n\nModel memprediksi karyawan ini **berisiko burnout**.")
            else:
                st.success(f"✅ **TIDAK BURNOUT**\n\nModel memprediksi karyawan ini **tidak berisiko burnout**.")

        with res_col2:
            st.metric("Probabilitas Burnout", f"{burnout_prob:.1f}%")
            st.progress(min(burnout_prob / 100, 1.0))

        with res_col3:
            color_map = {"Rendah": "✅", "Sedang": "⚠️", "Tinggi": "🔴"}
            st.metric("Risk Level", f"{color_map[risk_lvl]} {risk_lvl}")
            st.caption(f"Berdasarkan kombinasi stres, jam kerja, dan kepuasan")

        # Rekomendasi
        st.markdown("---")
        st.subheader("💡 Rekomendasi Intervensi")
        recs = []
        if work_hours >= 50:
            recs.append("🕐 **Kurangi jam kerja** – Pertimbangkan batasan kerja maksimal 45 jam/minggu.")
        if stress >= 7:
            recs.append("🧘 **Program manajemen stres** – Sediakan konseling atau sesi mindfulness.")
        if satisfaction <= 2.5:
            recs.append("😊 **Tingkatkan kepuasan kerja** – Evaluasi beban tugas, reward, dan work environment.")
        if remote < 20:
            recs.append("🏠 **Fleksibilitas remote** – Opsi kerja dari rumah bisa mengurangi kelelahan commuting.")
        if not recs:
            recs.append("✅ Kondisi karyawan ini relatif baik. Pertahankan keseimbangan saat ini.")
        for rec in recs:
            st.markdown(f"- {rec}")


# ═══════════════════════════════════════════════════════════════
# HALAMAN 4: PERFORMA MODEL
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Performa Model":
    st.title("📈 Performa & Evaluasi Model ML")
    st.markdown("---")

    st.subheader(f"🏆 Model Terpilih: {model_name}")
    st.markdown("Berikut adalah perbandingan semua model yang diuji:")

    st.dataframe(
        results_df[["Accuracy","F1","Precision","Recall","AUC-ROC"]].round(4),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Perbandingan Metrik")
        metrics = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        width = 0.18
        palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
        for i, (name, row) in enumerate(results_df.iterrows()):
            vals = [float(row[m]) for m in metrics]
            ax.bar(x + i*width, vals, width, label=name, color=palette[i])
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.set_title("Perbandingan Performa Model")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("💡 Interpretasi")
        st.markdown("""
        **Insight dari hasil evaluasi:**
        - Model berbasis tree (Decision Tree, Random Forest, Gradient Boosting)
          mencapai performa sempurna pada dataset ini.
        - Logistic Regression tetap memiliki AUC-ROC tinggi (~0.995), 
          menunjukkan kemampuan diskriminasi yang baik.
        - Fitur **StressLevel** dan **WorkHoursPerWeek** adalah prediktor terkuat.
        - Pada data nyata, Random Forest direkomendasikan karena lebih robust 
          terhadap overfitting dibanding Decision Tree tunggal.
        """)

    st.subheader("📌 Feature Importance")
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.barh(feat_imp["Feature"], feat_imp["Importance"],
                 color="#4CAF50")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Feature Importance Score")
        st.pyplot(fig2)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# HALAMAN 5: A/B TESTING
# ═══════════════════════════════════════════════════════════════
elif page == "🧪 A/B Testing":
    st.title("🧪 A/B Testing")
    st.markdown("---")

    from scipy import stats

    st.markdown("""
    ### Tujuan A/B Testing
    Memvalidasi secara statistik apakah perbedaan performa antar model 
    dan apakah beban kerja tinggi benar-benar mempengaruhi burnout secara signifikan.
    """)

    # Setup CV scores
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    @st.cache_data
    def compute_cv_scores():
        df_m = df.copy()
        le_g = LabelEncoder(); le_r = LabelEncoder()
        df_m["Gender_enc"]  = le_g.fit_transform(df_m["Gender"])
        df_m["JobRole_enc"] = le_r.fit_transform(df_m["JobRole"])
        df_m["StressWorkRatio"] = df_m["StressLevel"] / (df_m["WorkHoursPerWeek"] + 1)
        df_m["WorkLifeScore"]   = df_m["SatisfactionLevel"] * (1 - df_m["RemoteRatio"] / 100)
        df_m["HighRiskFlag"]    = ((df_m["StressLevel"] >= 7) & (df_m["WorkHoursPerWeek"] >= 50)).astype(int)
        df_m["SeniorEmployee"]  = (df_m["Experience"] >= 10).astype(int)
        FEATS = ["Age","Experience","WorkHoursPerWeek","RemoteRatio",
                 "SatisfactionLevel","StressLevel","Gender_enc","JobRole_enc",
                 "StressWorkRatio","WorkLifeScore","HighRiskFlag","SeniorEmployee"]
        X = StandardScaler().fit_transform(df_m[FEATS])
        y = df_m["Burnout"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X, y, cv=cv, scoring="f1")
        rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=cv, scoring="f1")
        gb = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42), X, y, cv=cv, scoring="f1")
        return lr, rf, gb

    with st.spinner("Menghitung CV scores..."):
        cv_lr, cv_rf, cv_gb = compute_cv_scores()

    tab1, tab2, tab3 = st.tabs(
        ["Test 1: LR vs RF", "Test 2: RF vs GB", "Test 3: Jam Kerja vs Burnout"]
    )

    with tab1:
        st.subheader("A/B Test 1: Logistic Regression (A) vs Random Forest (B)")
        t1, p1 = stats.ttest_ind(cv_lr, cv_rf)
        col1, col2 = st.columns(2)
        with col1:
            scores_data = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(5)],
                "LR F1": cv_lr.round(4),
                "RF F1": cv_rf.round(4)
            })
            st.dataframe(scores_data, use_container_width=True)
            st.metric("t-statistic", f"{t1:.4f}")
            st.metric("p-value", f"{p1:.6f}")
            if p1 < 0.05:
                st.error(f"✅ **TOLAK H0** – perbedaan signifikan (p < 0.05)\n\nPemenang: Random Forest")
            else:
                st.success("❌ Gagal tolak H0 – tidak ada perbedaan signifikan")
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            bp = ax.boxplot([cv_lr, cv_rf], labels=["LR", "RF"], patch_artist=True)
            colors = ["#2196F3", "#4CAF50"]
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_ylabel("F1 Score")
            ax.set_title("Distribusi CV F1: LR vs RF")
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("A/B Test 2: Random Forest (A) vs Gradient Boosting (B)")
        t2, p2 = stats.ttest_ind(cv_rf, cv_gb)
        col1, col2 = st.columns(2)
        with col1:
            scores_data2 = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(5)],
                "RF F1": cv_rf.round(4),
                "GB F1": cv_gb.round(4)
            })
            st.dataframe(scores_data2, use_container_width=True)
            st.metric("t-statistic", f"{t2:.4f}")
            st.metric("p-value", f"{p2:.6f}")
            if p2 < 0.05:
                st.error("✅ **TOLAK H0** – perbedaan signifikan")
            else:
                st.success("❌ Gagal tolak H0 – tidak ada perbedaan signifikan")
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            bp = ax.boxplot([cv_rf, cv_gb], labels=["RF", "GB"], patch_artist=True)
            colors = ["#4CAF50", "#FF9800"]
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_ylabel("F1 Score")
            ax.set_title("Distribusi CV F1: RF vs GB")
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.subheader("A/B Test 3: Burnout Rate – Jam Kerja Tinggi vs Rendah")
        threshold = st.slider("Batas jam kerja 'tinggi':", 40, 60, 50)
        low_grp  = df[df["WorkHoursPerWeek"] <  threshold]["Burnout"].values
        high_grp = df[df["WorkHoursPerWeek"] >= threshold]["Burnout"].values
        t3, p3 = stats.ttest_ind(low_grp, high_grp)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Burnout Rate (< {threshold} jam)", f"{low_grp.mean()*100:.2f}%",
                      delta=f"n={len(low_grp):,}")
            st.metric(f"Burnout Rate (≥ {threshold} jam)", f"{high_grp.mean()*100:.2f}%",
                      delta=f"n={len(high_grp):,}")
            st.metric("t-statistic", f"{t3:.4f}")
            st.metric("p-value", f"{p3:.6f}")
            if p3 < 0.05:
                st.error("✅ **TOLAK H0** – jam kerja tinggi berkontribusi signifikan terhadap burnout!")
            else:
                st.success("❌ Gagal tolak H0")
        with col2:
            rates = [low_grp.mean()*100, high_grp.mean()*100]
            labels = [f"< {threshold} jam\n(n={len(low_grp):,})",
                      f"≥ {threshold} jam\n(n={len(high_grp):,})"]
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(labels, rates, color=["#2196F3","#F44336"], edgecolor="black")
            ax.set_ylabel("Burnout Rate (%)")
            ax.set_title("Burnout Rate per Kelompok Jam Kerja")
            for bar, v in zip(bars, rates):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}%",
                        ha="center", fontsize=12)
            sig_txt = f"p = {p3:.4f} {'*' if p3<0.05 else '(ns)'}"
            ax.set_xlabel(sig_txt)
            st.pyplot(fig)
            plt.close()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "🔥 Burnout Risk Prediction System | Capstone Project – Data Science"
    "</div>",
    unsafe_allow_html=True
)
