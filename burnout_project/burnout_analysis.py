# ============================================================
#  BURNOUT RISK PREDICTION SYSTEM - DATA SCIENTIST PIPELINE
#  Capstone Project | Synthetic Employee Burnout Dataset
# ============================================================
# Tahapan:
#   1. Problem Discovery & Analisis Permasalahan
#   2. Data Wrangling (Gathering, Assessing, Cleaning)
#   3. Definisi Business Questions
#   4. Exploratory Data Analysis (EDA)
#   5. Visualisasi & Explanatory Analysis
#   6. Data Preparation untuk Model (Feature Engineering)
#   7. Model Building & Evaluation
#   8. A/B Testing
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score, precision_score, recall_score
)
from scipy import stats

# ─── Paths ───────────────────────────────────────────────────
DATA_PATH  = "/mnt/user-data/uploads/synthetic_employee_burnout.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="husl")
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

# ═══════════════════════════════════════════════════════════════
# BAGIAN 1 ─ PROBLEM DISCOVERY & ANALISIS PERMASALAHAN
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print(" BAGIAN 1: PROBLEM DISCOVERY & ANALISIS PERMASALAHAN")
print("=" * 65)

problem_statement = {
    "Masalah Utama": "Burnout karyawan menyebabkan penurunan produktivitas, "
                     "tingginya turnover, dan dampak kesehatan mental serius.",
    "Akar Penyebab Potensial": [
        "Beban kerja berlebih (WorkHoursPerWeek tinggi)",
        "Tingkat stres yang tidak terkelola (StressLevel tinggi)",
        "Kurangnya kepuasan kerja (SatisfactionLevel rendah)",
        "Ketidakseimbangan work-life (RemoteRatio tidak optimal)",
        "Faktor demografis (Usia, Pengalaman, Peran)",
    ],
    "Solusi yang Dikembangkan": "Burnout Risk Prediction System – model ML "
                                 "untuk mengklasifikasi risiko (Rendah/Sedang/Tinggi) "
                                 "beserta dashboard Streamlit interaktif.",
    "Stakeholder": ["HR Manager", "Tim Kesehatan Karyawan", "Manajemen Senior"],
    "Success Metric": "Model Accuracy ≥ 80%, AUC-ROC ≥ 0.85",
}

print("\n📋 Problem Statement:")
for k, v in problem_statement.items():
    if isinstance(v, list):
        print(f"\n  ▸ {k}:")
        for item in v: print(f"      - {item}")
    else:
        print(f"\n  ▸ {k}:\n      {v}")

print("\n✅ Problem Discovery selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 2 ─ DATA WRANGLING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 2: DATA WRANGLING")
print("=" * 65)

# ── 2a. Gathering ───────────────────────────────────────────
print("\n── 2a. Gathering Data ──")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset berhasil dimuat dari: {DATA_PATH}")
print(f"  Jumlah baris   : {df.shape[0]:,}")
print(f"  Jumlah kolom   : {df.shape[1]}")
print(f"\n  Kolom & tipe data:")
print(df.dtypes.to_string())
print(f"\n  5 baris pertama:")
print(df.head().to_string(index=False))

# ── 2b. Assessing ────────────────────────────────────────────
print("\n── 2b. Assessing Data ──")

missing = df.isnull().sum()
duplicates = df.duplicated().sum()
print(f"\n  Missing Values per kolom:")
print(missing.to_string())
print(f"\n  Jumlah baris duplikat  : {duplicates}")
print(f"\n  Statistik deskriptif:")
print(df.describe().round(2).to_string())

print(f"\n  Distribusi nilai target (Burnout):")
vc = df["Burnout"].value_counts()
for val, cnt in vc.items():
    print(f"    Burnout={val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

print(f"\n  Nilai unik kolom kategorikal:")
for col in ["Gender", "JobRole"]:
    print(f"    {col}: {df[col].unique().tolist()}")

# ── 2c. Cleaning ─────────────────────────────────────────────
print("\n── 2c. Cleaning Data ──")
df_clean = df.copy()

# Hapus duplikat jika ada
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"  Duplikat dihapus: {before - len(df_clean)} baris")

# Hapus kolom Name (tidak relevan untuk model)
df_clean = df_clean.drop(columns=["Name"])
print("  Kolom 'Name' dihapus (tidak relevan untuk model)")

# Imputasi missing value (jika ada)
num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
for col in num_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"  Imputasi numerik '{col}' → median")
for col in cat_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        print(f"  Imputasi kategorikal '{col}' → modus")

# Tambah kolom BurnoutLabel (multi-class risk level)
def risk_label(row):
    score = 0
    if row["WorkHoursPerWeek"] >= 55: score += 2
    elif row["WorkHoursPerWeek"] >= 45: score += 1
    if row["StressLevel"] >= 7: score += 2
    elif row["StressLevel"] >= 4: score += 1
    if row["SatisfactionLevel"] <= 2.0: score += 2
    elif row["SatisfactionLevel"] <= 3.5: score += 1
    if score >= 4: return "Tinggi"
    elif score >= 2: return "Sedang"
    else: return "Rendah"

df_clean["RiskLevel"] = df_clean.apply(risk_label, axis=1)
print("\n  Kolom 'RiskLevel' berhasil dibuat (Rendah / Sedang / Tinggi)")
print(f"  Distribusi RiskLevel:")
print(df_clean["RiskLevel"].value_counts().to_string())

print(f"\n  Dataset bersih: {df_clean.shape[0]:,} baris × {df_clean.shape[1]} kolom")
print("\n✅ Data Wrangling selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 3 ─ BUSINESS QUESTIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 3: DEFINISI BUSINESS QUESTIONS")
print("=" * 65)

bq_list = [
    ("BQ1", "Seberapa besar proporsi karyawan yang mengalami burnout?"),
    ("BQ2", "Apakah beban kerja (WorkHoursPerWeek) berkorelasi dengan burnout?"),
    ("BQ3", "Faktor apa yang paling berpengaruh terhadap risiko burnout?"),
    ("BQ4", "Bagaimana distribusi burnout antar JobRole?"),
    ("BQ5", "Apakah ada perbedaan burnout antara gender?"),
    ("BQ6", "Apakah kepuasan kerja rendah → stres tinggi → burnout?"),
    ("BQ7", "Bagaimana tingkat risiko (Rendah/Sedang/Tinggi) terdistribusi?"),
]

print("\n  Business Questions yang akan dijawab:")
for code, q in bq_list:
    print(f"  [{code}] {q}")

# ─── Jawab BQ secara ringkas ──────────────────────────────────
print("\n── Jawaban Ringkas Business Questions ──")

burnout_rate = df_clean["Burnout"].mean() * 100
print(f"\n[BQ1] Tingkat burnout keseluruhan: {burnout_rate:.1f}%")

corr_hours = df_clean["WorkHoursPerWeek"].corr(df_clean["Burnout"])
print(f"[BQ2] Korelasi WorkHoursPerWeek ↔ Burnout: {corr_hours:.3f}")

corr_stress = df_clean["StressLevel"].corr(df_clean["Burnout"])
print(f"[BQ6] Korelasi StressLevel ↔ Burnout: {corr_stress:.3f}")

print(f"\n[BQ4] Burnout rate per JobRole:")
br_role = df_clean.groupby("JobRole")["Burnout"].mean().sort_values(ascending=False) * 100
print(br_role.round(1).to_string())

print(f"\n[BQ5] Burnout rate per Gender:")
br_gender = df_clean.groupby("Gender")["Burnout"].mean() * 100
print(br_gender.round(1).to_string())

print(f"\n[BQ7] Distribusi RiskLevel:")
print((df_clean["RiskLevel"].value_counts(normalize=True) * 100).round(1).to_string())

print("\n✅ Business Questions selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 4 ─ EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)

# 4a. Korelasi matrix
num_df = df_clean.select_dtypes(include=np.number)
corr_matrix = num_df.corr()
print("\n── Korelasi Matrix (numerik) ──")
print(corr_matrix.round(3).to_string())

# 4b. Outlier deteksi (IQR)
print("\n── Deteksi Outlier (IQR Method) ──")
outlier_summary = {}
for col in ["WorkHoursPerWeek", "StressLevel", "SatisfactionLevel", "Age", "Experience"]:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_summary[col] = n_out
    print(f"  {col:<25}: {n_out} outlier  (batas: [{lower:.2f}, {upper:.2f}])")

# 4c. Statistik group burnout vs non-burnout
print("\n── Statistik Burnout vs Non-Burnout ──")
group_stats = df_clean.groupby("Burnout")[
    ["WorkHoursPerWeek", "StressLevel", "SatisfactionLevel", "Age"]
].mean().round(2)
print(group_stats.to_string())

print("\n✅ EDA selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 5 ─ VISUALISASI & EXPLANATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 5: VISUALISASI & EXPLANATORY ANALYSIS")
print("=" * 65)

# ── Figure 1: Overview distribusi ────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Distribusi Fitur Utama Dataset Burnout Karyawan",
              fontsize=16, fontweight="bold", y=1.01)

features = ["Age", "WorkHoursPerWeek", "StressLevel",
            "SatisfactionLevel", "Experience", "RemoteRatio"]
for ax, col in zip(axes.flat, features):
    sns.histplot(data=df_clean, x=col, hue="Burnout",
                 bins=20, kde=True, ax=ax,
                 palette={0: PALETTE[0], 1: PALETTE[1]})
    ax.set_title(f"Distribusi {col}")
    ax.set_xlabel(col)
    handles = ax.get_legend().legend_handles if ax.get_legend() else []
    if handles:
        ax.legend(handles, ["Tidak Burnout", "Burnout"], title="Status")

plt.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/fig1_distribusi_fitur.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig1_distribusi_fitur.png disimpan")

# ── Figure 2: Heatmap korelasi ────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
ax.set_title("Heatmap Korelasi Antar Fitur Numerik", fontsize=14, fontweight="bold")
plt.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/fig2_heatmap_korelasi.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig2_heatmap_korelasi.png disimpan")

# ── Figure 3: Burnout per JobRole & Gender ───────────────────
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Burnout Rate per Kategori", fontsize=14, fontweight="bold")

# BQ4 – per JobRole
br_role_sorted = br_role.sort_values()
bars = axes[0].barh(br_role_sorted.index, br_role_sorted.values,
                    color=PALETTE[:len(br_role_sorted)])
axes[0].set_xlabel("Burnout Rate (%)")
axes[0].set_title("Burnout Rate per JobRole")
for bar, val in zip(bars, br_role_sorted.values):
    axes[0].text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=10)

# BQ5 – per Gender
gender_burnout = df_clean.groupby("Gender")["Burnout"].mean() * 100
axes[1].bar(gender_burnout.index, gender_burnout.values,
            color=[PALETTE[0], PALETTE[1]], edgecolor="black")
axes[1].set_ylabel("Burnout Rate (%)")
axes[1].set_title("Burnout Rate per Gender")
for i, (g, val) in enumerate(gender_burnout.items()):
    axes[1].text(i, val + 0.2, f"{val:.1f}%", ha="center", fontsize=12)

plt.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/fig3_burnout_kategori.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig3_burnout_kategori.png disimpan")

# ── Figure 4: Boxplot Stress & WorkHours vs Burnout ──────────
fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle("Perbandingan Beban Kerja & Stres: Burnout vs Tidak",
              fontsize=14, fontweight="bold")

for ax, col, title in zip(axes,
                           ["WorkHoursPerWeek", "StressLevel"],
                           ["Jam Kerja per Minggu", "Tingkat Stres"]):
    df_box = df_clean.copy()
    df_box["BurnoutLabel"] = df_box["Burnout"].map({0: "Tidak Burnout", 1: "Burnout"})
    sns.boxplot(data=df_box, x="BurnoutLabel", y=col,
                palette={"Tidak Burnout": PALETTE[0], "Burnout": PALETTE[1]}, ax=ax)
    ax.set_xlabel("")
    ax.set_title(title)
    ax.set_xlabel("")

plt.tight_layout()
fig4.savefig(f"{OUTPUT_DIR}/fig4_boxplot_burnout.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig4_boxplot_burnout.png disimpan")

# ── Figure 5: Risk Level Distribution ────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle("Distribusi Tingkat Risiko Burnout", fontsize=14, fontweight="bold")

risk_counts = df_clean["RiskLevel"].value_counts()
order = ["Rendah", "Sedang", "Tinggi"]
colors_risk = ["#4CAF50", "#FF9800", "#F44336"]

axes[0].pie([risk_counts.get(r, 0) for r in order],
            labels=order, colors=colors_risk, autopct="%1.1f%%",
            startangle=140, textprops={"fontsize": 12})
axes[0].set_title("Proporsi RiskLevel")

axes[1].bar(order, [risk_counts.get(r, 0) for r in order],
            color=colors_risk, edgecolor="black")
axes[1].set_ylabel("Jumlah Karyawan")
axes[1].set_title("Jumlah per RiskLevel")
for i, r in enumerate(order):
    val = risk_counts.get(r, 0)
    axes[1].text(i, val + 10, str(val), ha="center", fontsize=12)

plt.tight_layout()
fig5.savefig(f"{OUTPUT_DIR}/fig5_risk_level.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig5_risk_level.png disimpan")

# ── Figure 6: Scatter WorkHours vs Stress colored by Burnout ─
fig6, ax = plt.subplots(figsize=(10, 7))
colors_scatter = df_clean["Burnout"].map({0: PALETTE[0], 1: PALETTE[1]})
scatter = ax.scatter(df_clean["WorkHoursPerWeek"], df_clean["StressLevel"],
                     c=colors_scatter, alpha=0.5, s=30, edgecolors="none")
ax.set_xlabel("Jam Kerja per Minggu", fontsize=12)
ax.set_ylabel("Tingkat Stres", fontsize=12)
ax.set_title("Hubungan Beban Kerja vs Stres berdasarkan Status Burnout",
             fontsize=13, fontweight="bold")
legend_elements = [
    plt.scatter([], [], c=PALETTE[0], label="Tidak Burnout"),
    plt.scatter([], [], c=PALETTE[1], label="Burnout"),
]
ax.legend(handles=[
    plt.Line2D([0],[0], marker="o", color="w",
               markerfacecolor=PALETTE[0], markersize=10, label="Tidak Burnout"),
    plt.Line2D([0],[0], marker="o", color="w",
               markerfacecolor=PALETTE[1], markersize=10, label="Burnout"),
])
plt.tight_layout()
fig6.savefig(f"{OUTPUT_DIR}/fig6_scatter_burnout.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig6_scatter_burnout.png disimpan")

print("\n✅ Visualisasi selesai – 6 gambar tersimpan.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 6 ─ FEATURE ENGINEERING & DATA PREP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 6: FEATURE ENGINEERING & DATA DICTIONARY")
print("=" * 65)

df_model = df_clean.copy()

# Encoding kategorikal
le_gender  = LabelEncoder()
le_role    = LabelEncoder()
df_model["Gender_enc"]  = le_gender.fit_transform(df_model["Gender"])
df_model["JobRole_enc"] = le_role.fit_transform(df_model["JobRole"])
print("  ✓ Encoding Gender & JobRole selesai")
print(f"    Gender mapping  : {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"    JobRole mapping : {dict(zip(le_role.classes_, le_role.transform(le_role.classes_)))}")

# Fitur rekayasa
df_model["StressWorkRatio"]    = df_model["StressLevel"] / (df_model["WorkHoursPerWeek"] + 1)
df_model["WorkLifeScore"]      = df_model["SatisfactionLevel"] * (1 - df_model["RemoteRatio"] / 100)
df_model["HighRiskFlag"]       = ((df_model["StressLevel"] >= 7) &
                                   (df_model["WorkHoursPerWeek"] >= 50)).astype(int)
df_model["SeniorEmployee"]     = (df_model["Experience"] >= 10).astype(int)
print("\n  ✓ Fitur baru dibuat:")
print("    - StressWorkRatio    : StressLevel / (WorkHoursPerWeek + 1)")
print("    - WorkLifeScore      : SatisfactionLevel * (1 - RemoteRatio/100)")
print("    - HighRiskFlag       : 1 jika StressLevel≥7 & WorkHours≥50")
print("    - SeniorEmployee     : 1 jika Experience≥10")

# Pilih fitur akhir
FEATURES = [
    "Age", "Experience", "WorkHoursPerWeek", "RemoteRatio",
    "SatisfactionLevel", "StressLevel",
    "Gender_enc", "JobRole_enc",
    "StressWorkRatio", "WorkLifeScore", "HighRiskFlag", "SeniorEmployee"
]
TARGET = "Burnout"

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"\n  Fitur yang digunakan ({len(FEATURES)}):")
for f in FEATURES: print(f"    - {f}")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n  Train set : {X_train.shape[0]:,} baris ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test  set : {X_test.shape[0]:,} baris ({len(X_test)/len(X)*100:.0f}%)")

# Data Dictionary
data_dict = pd.DataFrame({
    "Kolom": [
        "Age", "Gender", "JobRole", "Experience", "WorkHoursPerWeek",
        "RemoteRatio", "SatisfactionLevel", "StressLevel", "Burnout",
        "RiskLevel", "StressWorkRatio", "WorkLifeScore",
        "HighRiskFlag", "SeniorEmployee"
    ],
    "Tipe": [
        "int", "str", "str", "int", "int",
        "int", "float", "int", "int (0/1)",
        "str (Rendah/Sedang/Tinggi)", "float", "float",
        "int (0/1)", "int (0/1)"
    ],
    "Deskripsi": [
        "Usia karyawan (tahun)",
        "Jenis kelamin (Male/Female)",
        "Posisi jabatan",
        "Lama pengalaman kerja (tahun)",
        "Rata-rata jam kerja per minggu",
        "Persentase kerja remote (%)",
        "Skor kepuasan kerja (1–5)",
        "Tingkat stres (1–10)",
        "Label target burnout (0=Tidak, 1=Ya)",
        "Label risiko multi-kelas",
        "Rasio stres terhadap beban kerja",
        "Skor keseimbangan kerja-kehidupan",
        "Flag risiko tinggi (stres & jam kerja ekstrem)",
        "Flag karyawan senior (≥10 tahun)"
    ]
})
print("\n── Data Dictionary ──")
print(data_dict.to_string(index=False))
data_dict.to_csv(f"{OUTPUT_DIR}/data_dictionary.csv", index=False)
print("\n  ✓ Data Dictionary disimpan ke data_dictionary.csv")

print("\n✅ Feature Engineering selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 7 ─ MODEL BUILDING & EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 7: MODEL BUILDING & EVALUATION")
print("=" * 65)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n  Training & Evaluasi 4 Model:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_scaled_df, y, cv=cv, scoring="f1")

    results[name] = {
        "Accuracy": acc, "F1": f1, "Precision": prec,
        "Recall": rec, "AUC-ROC": auc,
        "CV F1 Mean": cv_scores.mean(), "CV F1 Std": cv_scores.std()
    }
    print(f"  ── {name} ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     AUC-ROC   : {auc:.4f}")
    print(f"     CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# Rangkuman performa
results_df = pd.DataFrame(results).T.round(4)
print("── Rangkuman Performa Model ──")
print(results_df.to_string())

best_model_name = results_df["AUC-ROC"].idxmax()
best_model      = models[best_model_name]
print(f"\n  🏆 Model terbaik (AUC-ROC): {best_model_name}")

# Classification Report model terbaik
y_pred_best = best_model.predict(X_test)
print(f"\n  Classification Report – {best_model_name}:")
print(classification_report(y_test, y_pred_best,
                             target_names=["Tidak Burnout", "Burnout"]))

# ── Figure 7: Perbandingan model ──────────────────────────────
fig7, axes = plt.subplots(1, 2, figsize=(16, 6))
fig7.suptitle("Perbandingan Performa Model ML", fontsize=14, fontweight="bold")

metrics = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
x = np.arange(len(metrics))
width = 0.18
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    axes[0].bar(x + i*width, vals, width, label=name, color=PALETTE[i])
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Score")
axes[0].set_title("Metrik Evaluasi per Model")
axes[0].legend(fontsize=8)

# Confusion matrix model terbaik
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Tidak Burnout", "Burnout"],
            yticklabels=["Tidak Burnout", "Burnout"])
axes[1].set_title(f"Confusion Matrix – {best_model_name}")
axes[1].set_ylabel("Aktual")
axes[1].set_xlabel("Prediksi")

plt.tight_layout()
fig7.savefig(f"{OUTPUT_DIR}/fig7_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig7_model_comparison.png disimpan")

# ── Figure 8: Feature Importance & ROC Curve ─────────────────
fig8, axes = plt.subplots(1, 2, figsize=(16, 7))
fig8.suptitle(f"Feature Importance & ROC Curve – {best_model_name}",
              fontsize=14, fontweight="bold")

importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})\
    .sort_values("Importance", ascending=True)
axes[0].barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
             color=PALETTE[2])
axes[0].set_title("Feature Importance")
axes[0].set_xlabel("Importance Score")

y_proba_best = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
auc_val = roc_auc_score(y_test, y_proba_best)
axes[1].plot(fpr, tpr, color=PALETTE[1], lw=2,
             label=f"AUC = {auc_val:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
fig8.savefig(f"{OUTPUT_DIR}/fig8_feature_roc.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ✓ fig8_feature_roc.png disimpan")

print("\n✅ Model Building & Evaluation selesai.")

# ═══════════════════════════════════════════════════════════════
# BAGIAN 8 ─ A/B TESTING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" BAGIAN 8: A/B TESTING DENGAN PYTHON")
print("=" * 65)

print("""
  Hipotesis A/B Testing:
  ─────────────────────
  Eksperimen : Penggunaan model Random Forest vs Logistic Regression
               untuk memberikan intervensi berbasis prediksi burnout.

  H0 (Null)      : Tidak ada perbedaan signifikan performa antara
                   model A (Logistic Regression) dan B (Random Forest)
                   dalam hal F1-Score.
  H1 (Alternatif): Model B (Random Forest) memiliki F1-Score yang
                   secara signifikan lebih tinggi dari model A.

  Metode   : Two-Sample t-test pada distribusi CV scores (k-fold).
  Threshold: α = 0.05
""")

# Ambil CV F1 scores per model menggunakan cross_val_score
cv_scores_lr = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X_scaled_df, y, cv=cv, scoring="f1"
)
cv_scores_rf = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_scaled_df, y, cv=cv, scoring="f1"
)
cv_scores_gb = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_scaled_df, y, cv=cv, scoring="f1"
)

print("  CV F1 Scores (5-fold) per Model:")
print(f"    Logistic Regression : {cv_scores_lr.round(4).tolist()}")
print(f"    Random Forest       : {cv_scores_rf.round(4).tolist()}")
print(f"    Gradient Boosting   : {cv_scores_gb.round(4).tolist()}")

print(f"\n  Mean ± Std:")
print(f"    LR  : {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print(f"    RF  : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"    GB  : {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")

# ── A/B Test 1: LR vs RF ─────────────────────────────────────
t_stat, p_val = stats.ttest_ind(cv_scores_lr, cv_scores_rf)
print(f"\n── A/B Test 1: Logistic Regression (A) vs Random Forest (B) ──")
print(f"   t-statistic : {t_stat:.4f}")
print(f"   p-value     : {p_val:.4f}")
if p_val < 0.05:
    winner = "Random Forest" if cv_scores_rf.mean() > cv_scores_lr.mean() else "Logistic Regression"
    print(f"   Kesimpulan  : ✅ TOLAK H0 – perbedaan signifikan (p < 0.05)")
    print(f"   Pemenang    : {winner}")
else:
    print(f"   Kesimpulan  : ❌ GAGAL tolak H0 – tidak ada perbedaan signifikan (p ≥ 0.05)")

# ── A/B Test 2: RF vs GB ─────────────────────────────────────
t_stat2, p_val2 = stats.ttest_ind(cv_scores_rf, cv_scores_gb)
print(f"\n── A/B Test 2: Random Forest (A) vs Gradient Boosting (B) ──")
print(f"   t-statistic : {t_stat2:.4f}")
print(f"   p-value     : {p_val2:.4f}")
if p_val2 < 0.05:
    winner2 = "Gradient Boosting" if cv_scores_gb.mean() > cv_scores_rf.mean() else "Random Forest"
    print(f"   Kesimpulan  : ✅ TOLAK H0 – perbedaan signifikan (p < 0.05)")
    print(f"   Pemenang    : {winner2}")
else:
    print(f"   Kesimpulan  : ❌ GAGAL tolak H0 – tidak ada perbedaan signifikan (p ≥ 0.05)")

# ── A/B Test 3: Burnout rate tinggi jam kerja vs rendah ──────
high_hours = df_clean[df_clean["WorkHoursPerWeek"] >= 50]["Burnout"].values
low_hours  = df_clean[df_clean["WorkHoursPerWeek"] < 40]["Burnout"].values
t_stat3, p_val3 = stats.ttest_ind(high_hours, low_hours)
print(f"\n── A/B Test 3: Burnout Rate – Jam Kerja Tinggi vs Rendah ──")
print(f"   Burnout rate (≥50 jam) : {high_hours.mean()*100:.2f}%  (n={len(high_hours):,})")
print(f"   Burnout rate (<40 jam) : {low_hours.mean()*100:.2f}%  (n={len(low_hours):,})")
print(f"   t-statistic : {t_stat3:.4f}")
print(f"   p-value     : {p_val3:.4f}")
if p_val3 < 0.05:
    print(f"   Kesimpulan  : ✅ TOLAK H0 – karyawan jam kerja tinggi memiliki")
    print(f"                 burnout rate yang berbeda secara signifikan")
else:
    print(f"   Kesimpulan  : ❌ GAGAL tolak H0")

# ── Figure 9: A/B Testing Visualisasi ────────────────────────
fig9, axes = plt.subplots(1, 3, figsize=(18, 6))
fig9.suptitle("A/B Testing – Visualisasi Perbandingan", fontsize=14, fontweight="bold")

# Plot 1: CV F1 distribusi
model_names_ab = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
scores_ab = [cv_scores_lr, cv_scores_rf, cv_scores_gb]
bp = axes[0].boxplot(scores_ab, labels=model_names_ab, patch_artist=True,
                     boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
for patch, color in zip(bp["boxes"], PALETTE[:3]):
    patch.set_facecolor(color)
axes[0].set_ylabel("F1 Score (CV)")
axes[0].set_title("Distribusi CV F1-Score per Model")
axes[0].yaxis.grid(True)

# Plot 2: Mean F1 dengan error bar
means = [s.mean() for s in scores_ab]
stds  = [s.std() for s in scores_ab]
axes[1].bar(model_names_ab, means, yerr=stds,
            color=PALETTE[:3], capsize=6, edgecolor="black")
axes[1].set_ylabel("Mean F1 Score")
axes[1].set_title("Mean F1 Score ± Std Dev")
axes[1].set_ylim(0, max(means) * 1.3)
for i, (m, s) in enumerate(zip(means, stds)):
    axes[1].text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=11)

# Plot 3: Burnout rate per kelompok jam kerja
jam_groups = ["< 40 jam\n(n={:,})".format(len(low_hours)),
              "≥ 50 jam\n(n={:,})".format(len(high_hours))]
burnout_rates = [low_hours.mean() * 100, high_hours.mean() * 100]
bars3 = axes[2].bar(jam_groups, burnout_rates,
                    color=[PALETTE[0], PALETTE[1]], edgecolor="black")
axes[2].set_ylabel("Burnout Rate (%)")
axes[2].set_title(f"Burnout Rate: Jam Kerja\n(p-value = {p_val3:.4f})")
for bar, val in zip(bars3, burnout_rates):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.2,
                 f"{val:.2f}%", ha="center", fontsize=11)
sig_label = "* Signifikan (p<0.05)" if p_val3 < 0.05 else "Tidak Signifikan"
axes[2].text(0.5, max(burnout_rates)*1.15, sig_label,
             ha="center", transform=axes[2].transAxes,
             fontsize=11, color="red" if p_val3 < 0.05 else "gray")

plt.tight_layout()
fig9.savefig(f"{OUTPUT_DIR}/fig9_ab_testing.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n  ✓ fig9_ab_testing.png disimpan")

print("\n✅ A/B Testing selesai.")

# ═══════════════════════════════════════════════════════════════
# SIMPAN MODEL UNTUK STREAMLIT
# ═══════════════════════════════════════════════════════════════
import pickle

model_artifacts = {
    "best_model"       : best_model,
    "best_model_name"  : best_model_name,
    "scaler"           : scaler,
    "le_gender"        : le_gender,
    "le_role"          : le_role,
    "features"         : FEATURES,
    "results_df"       : results_df.to_dict(),
}
with open(f"{OUTPUT_DIR}/model_artifacts.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)
print(f"\n  ✓ model_artifacts.pkl disimpan ke {OUTPUT_DIR}/")

df_clean.to_csv(f"{OUTPUT_DIR}/df_clean.csv", index=False)
print(f"  ✓ df_clean.csv disimpan ke {OUTPUT_DIR}/")

# ─── Ringkasan Akhir ─────────────────────────────────────────
print("\n" + "=" * 65)
print(" ✅  PIPELINE DATA SCIENTIST SELESAI SELURUHNYA")
print("=" * 65)
print(f"""
  File yang dihasilkan:
    📊 fig1_distribusi_fitur.png
    🔥 fig2_heatmap_korelasi.png
    📊 fig3_burnout_kategori.png
    📦 fig4_boxplot_burnout.png
    🎯 fig5_risk_level.png
    🔵 fig6_scatter_burnout.png
    🤖 fig7_model_comparison.png
    📈 fig8_feature_roc.png
    🧪 fig9_ab_testing.png
    📄 data_dictionary.csv
    🤖 model_artifacts.pkl
    📋 df_clean.csv
""")
