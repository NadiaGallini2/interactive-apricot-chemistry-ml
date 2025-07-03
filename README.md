# Phenolic Compounds Analysis in Grapevine

**Interactive Streamlit application for visualization and statistical analysis of phenolic compounds**
Features include correlation heatmaps, distribution histograms, regression modeling, SHAP interpretability, ANOVA, and Multiple Factor Analysis (MFA).

---

## 📂 Project Structure

```
.
├── app.py             # Main Streamlit script
├── README.md          # This file
├── requirements.txt   # Python dependencies with exact versions
└── plots/             # Automatically generated figures
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```
   git clone https://github.com/your-username/phenolics-streamlit.git
   cd phenolics-streamlit
   ```

2. **Create a virtual environment**

   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - **Windows (CMD)**

     ```
     venv\Scripts\activate
     ```

   - **Windows (PowerShell)**

     ```powershell
     . .\venv\Scripts\Activate.ps1
     ```

   - **macOS/Linux**

     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

Once running, open the local URL provided by Streamlit (usually `http://localhost:8501`) in your browser.

---

## 📝 Dependencies

All packages are pinned to ensure reproducibility. See `requirements.txt` for exact versions:

```
pandas==1.3.5
numpy==1.21.6
matplotlib==3.5.1
seaborn==0.11.2
streamlit==1.18.1
scikit-learn==1.0.2
scipy==1.7.3
statsmodels==0.12.2
shap==0.40.0
prince==0.14.0
openpyxl==3.1.2
```

---

## 🎛️ Application Controls

Use the sidebar to configure:

- **Select Dataset**: choose your input CSV or Excel file
- **Number of Synthetic Observations**: per cultivar/compound
- **Noise Level**: relative standard deviation for Gaussian noise
- **Variety Shift**: maximum uniform shift applied to group means

Results and figures will update interactively.

---

## 🗂️ Output Files

Generated plots are saved to the `plots/` directory, for example:

- `plots/correlation_heatmap.png`
- `plots/<Compound>_distribution.png`
- `plots/ANOVA_results.csv`
- `plots/MFA_biplot.png`

---

## 📝 Patents

This application (“Interactive Machine Learning Platform for Phenolic Compound Analysis”) is officially registered with the Federal Service for Intellectual Property (Rospatent) as a computer program under registration number **2025661422** (May 6, 2025), confirming its novelty and practical readiness.

---

## ⚖️ License

**© 2025 Nadia Gallini. All rights reserved.**
No part of this software may be reproduced, distributed, or transmitted in any form or by any means without the prior written permission of the author. Recreational or research use within your organization is permitted; commercial distribution or modification requires explicit consent.

---

_Author: Nadia Gallini_
_Contact: [gallini.nadi@yandex.ru](mailto:gallini.nadi@yandex.ru)_
