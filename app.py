import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import prince
import scipy.stats as stats
import os
import shutil

import random
import numpy as np
random.seed(42)
np.random.seed(42)

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Create (or clear) the folder for saving plots
plots_dir = "plots"
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir, exist_ok=True)

@st.cache_data
def load_chemical_data():
    data_2018 = {
        "Variety": [
            "Aldebar", 
            "Artek Noviy", 
            "Vozrojdenije", 
            "Zvezdochet", 
            "Zevs", 
            "Iskorka Tavridyi", 
            "Krymskiy Amur (st.)", 
            "Samarityanin", 
            "Fiolent", 
            "Frigat", 
            "Shalard 2", 
            "Shedevr"
        ],
        "Year": [2018] * 12,
        "Dry_Matter": [18.20, 14.20, 17.00, 15.60, 24.90, 15.90, 20.45, 16.65, 16.80, 17.05, 16.70, 19.20],
        "Monosaccharides": [3.20, 0.91, 0.98, 1.14, 3.89, 1.97, 1.69, 2.38, 1.14, 2.52, 1.14, 1.28],
        "Sum_Sugars": [12.06, 3.94, 6.86, 3.38, 9.60, 4.21, 6.41, 6.86, 7.78, 11.57, 5.50, 6.86],
        "Titratable_Acids": [0.99, 1.05, 0.42, 1.39, 1.77, 1.05, 0.88, 1.82, 0.91, 0.95, 0.54, 0.91],
        "Ascorbic_Acid": [11.44, 6.78, 8.27, 15.84, 8.45, 11.44, 8.10, 6.16, 6.86, 6.51, 18.48, 13.20],
        "Leukoanthocyanins": [164.0, 28.0, 128.0, 312.0, 184.0, 408.0, 136.0, 480.0, 240.0, 56.0, 40.0, 72.0],
        "Water_Soluble_Pectin": [0.65, 0.48, 0.63, 0.65, 0.57, 0.57, 0.63, 0.52, 0.70, 0.67, 0.44, 0.69],
        "Protopectin": [0.85, 0.56, 0.52, 0.54, 0.78, 0.48, 0.53, 0.74, 0.61, 0.52, 0.46, 0.53],
        "Sum_Pectin": [1.50, 1.04, 1.15, 1.19, 1.35, 1.05, 1.16, 1.26, 1.31, 1.19, 0.90, 1.22]
    }
    return pd.DataFrame(data_2018)

# Load the original data
df = load_chemical_data()
st.title("Analysis of Fruit Chemical Composition")

st.markdown("""
## Interactive ML Platform for Apricot Fruit Chemistry

**Purpose:** Developed to accompany the research article  
*“Comprehensive Analysis of the Chemical Composition of Apricot Fruits Using Synthetic Data Generation and Machine Learning Methods.”*

**Patent:** Registered with Rospatent (Reg. No. 2025661422, May 6, 2025)

---

### Method Overview
- **Synthetic Data Generation** Structured mean shifts + Gaussian noise to augment limited datasets  
- **Correlation Analysis** Pearson correlations reveal linear dependencies  
- **ANOVA** One-way tests for cultivar differences  
- **Random Forest Regression** Predicts dry matter content (R² metric)  
- **SHAP Analysis** Shapley values explain feature contributions  
- **Multiple Factor Analysis (MFA)** Visualizes multivariate clustering  
""")


# --- Parameters for generating synthetic data ---
st.sidebar.header("Synthetic Data Generation Parameters")
num_synthetic = st.sidebar.slider(
    "Number of synthetic observations (per variety)", 
    min_value=5, 
    max_value=1000, 
    value=100, 
    step=50
)
noise_level = st.sidebar.slider(
    "Noise level", 
    min_value=0.01, 
    max_value=2.0, 
    value=0.2, 
    step=0.01
)
variety_shift = st.sidebar.slider(
    "Maximum shift for varieties", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.0, 
    step=0.5,
    help="A systematic random shift to enhance differences between varieties."
)
with st.sidebar.expander("Authors", expanded=False):
    st.markdown("- **Nadezhda I. Gallini** — [GitHub](https://github.com/NadiaGallini2) · [Email](mailto:your.email@example.com)")
    st.markdown("- **Anatoly N. Kazak**")
    st.markdown("- **Viktor I. Gallini**")
    st.markdown("- **Vadim V. Korzin**")
    st.markdown("- **Yuri V. Grishin**")

# --- Generate synthetic data ---
numeric_cols = df.select_dtypes(include=np.number).columns

# Compute average values by variety to have a baseline for each variety
grouped_means = df.groupby("Variety")[numeric_cols].mean()

synthetic_rows = []
for idx, row in df.iterrows():
    variety_name = row["Variety"]
    # For the given variety, select a random shift (± variety_shift) for each feature
    random_shifts = np.random.uniform(-variety_shift, variety_shift, size=len(numeric_cols))
    
    for _ in range(num_synthetic):
        syn_row = row.copy()
        for i, col in enumerate(numeric_cols):
            # Standard deviation of the original feature
            col_std = df[col].std()
            # Shift for the given variety (same for the whole set)
            shift_for_variety = random_shifts[i]
            # Add systematic shift + noise
            syn_row[col] = grouped_means.loc[variety_name, col] + shift_for_variety
            syn_row[col] += np.random.normal(0, noise_level * col_std)
        synthetic_rows.append(syn_row)

df_augmented = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

st.subheader("Data after generating synthetic observations")
st.dataframe(df_augmented.head(20))

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
# corr_matrix = df_augmented.select_dtypes(include=np.number).corr()
corr_matrix = df_augmented.select_dtypes(include=np.number).drop("Year", axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)
heatmap_filename = os.path.join(plots_dir, "correlation_heatmap_eng.png")
fig.savefig(heatmap_filename, format='png', dpi=150)
plt.close(fig)

# Сохранение матрицы корреляций в Excel
excel_filename = os.path.join(plots_dir, "correlation_matrix.xlsx")
corr_matrix.to_excel(excel_filename, index=True)

# Function to format p-value
def format_p_value(p):
    if p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.3f}"

# --- ANOVA Analysis by Variety ---
st.subheader("ANOVA Analysis by Variety")
p_values = {}
features_for_anova = ["Dry_Matter", "Monosaccharides", "Sum_Sugars"]
for col in features_for_anova:
    groups = [group[col].values for _, group in df_augmented.groupby("Variety")]
    f_val, p_val = stats.f_oneway(*groups)
    p_values[col] = format_p_value(p_val)

anova_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p-value'])
st.dataframe(anova_df)
anova_df.to_csv(os.path.join(plots_dir, "anova_results.csv"))

# --- Random Forest: Prediction of Dry Matter Content ---
st.subheader("Random Forest: Prediction of Dry Matter Content")

# (1) Numeric columns (excluding the target)
all_numeric = df_augmented.select_dtypes(include=np.number).columns.tolist()
all_numeric.remove("Dry_Matter")
X_num = df_augmented[all_numeric]

# (2) Categorical (Year, Variety)
X_cat = pd.get_dummies(df_augmented[["Year", "Variety"]], drop_first=True)

# Combine the features
X = pd.concat([X_num, X_cat], axis=1)
y = df_augmented["Dry_Matter"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.write(f"R² score: {r2:.3f}")

# --- SHAP Analysis ---
st.subheader("SHAP Analysis")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
shap_fig = plt.gcf()
st.pyplot(shap_fig)
shap_filename = os.path.join(plots_dir, "shap_summary.png")
shap_fig.savefig(shap_filename, format='png', dpi=150)
plt.clf()

# --- MFA Analysis by Variety ---
st.subheader("MFA Analysis by Variety")
df_mfa = df_augmented.select_dtypes(include=np.number).copy()
df_mfa.columns = pd.MultiIndex.from_tuples([("Chemical", col) for col in df_mfa.columns])

mfa = prince.MFA(n_components=2, random_state=42)
mfa = mfa.fit(df_mfa, groups={'Chemical': list(df_mfa.columns)})
mfa_coords = mfa.row_coordinates(df_mfa)

unique_varieties = df_augmented["Variety"].unique()

# Ручная палитра: 12 ярких уникальных цветов
distinct_palette = [
    "#e41a1c",  # Aldebar (red)
    "#ff7f00",  # Artek Noviy (orange)
    "#fdfd00",  # Vozrojdenije (yellow)
    "#4daf4a",  # Zvezdochet (green)
    "#00b2d4",  # Zevs (turquoise)
    "#377eb8",  # Iskorka Tavridyi (blue)
    "#984ea3",  # Krymskiy Amur (st.) (purple)
    "#a65628",  # Samarityanin (brown)
    "#f781bf",  # Fiolent (pink)
    "#999999",  # Frigat (gray)
    "#1b9e77",  # Shalard 2 (green-turquoise)
    "#d95f02",  # Shedevr (dark orange)
]
palette_dict = dict(zip(unique_varieties, distinct_palette))

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    x=mfa_coords[0],
    y=mfa_coords[1],
    hue=df_augmented["Variety"],
    palette=palette_dict,
    s=80,
    alpha=0.85,
    ax=ax
)
ax.set_title("MFA Analysis by Variety (Distinct Colors)")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.legend(title="Variety", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)
mfa_filename = os.path.join(plots_dir, "mfa_analysis.png")
fig.savefig(mfa_filename, format='png', dpi=150)
plt.close(fig)

st.markdown("---")
# st.markdown("### Authors")
# st.markdown("""
# - **Nadezhda I. Gallini** — [GitHub](https://github.com/NadiaGallini2) · [Email](mailto:gallini.nadi@yandex.ru)  
# - **Anatoly N. Kazak**  
# - **Viktor I. Gallini**  
# - **Vadim V. Korzin**  
# - **Yuri V. Grishin**
# """)

