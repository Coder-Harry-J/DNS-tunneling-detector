# 🛡️ DNS Tunneling Detector — Network Intrusion Detection System (NIDS)

An enterprise-grade, 12-stage Machine Learning Network Intrusion Detection System (NIDS) engineered to detect stealthy DNS tunneling, C2 beacons, and data exfiltration in real-time, benchmarked against the **CIC-Bell-DNS-EXF-2021** dataset.

Includes an interactive **macOS-glassmorphic Flask deployment dashboard** comparing three distinct machine learning paradigms simultaneously: **CART (Decision Tree)**, **SVM (RBF Kernel)**, and **ANN (Multilayer Perceptron)**.

---

## 🚀 Key Highlights & Architectural Metrics

Evaluated on the authentic holdout test set (**34,071 queries with an authentic 0.25% attack base rate**):

| Model | Status | Test Accuracy | Precision | Recall (Catch Rate) | F1-Score | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CART** | *Pre-Tuned Base* | $99.10\%$ | $7.66\%$ | $23.53\%$ | `0.1156` | $0.6311$ | $0.0175$ |
| **CART** | **Final Tuned** | **$99.19\%$** | **$16.96\%$** | **$57.65\%$** | **`0.2620`** | **$0.8866$** | **$0.1313$** |
| *CART Gain* | *GridSearchCV* | *+0.09%* | **+121.4%** | **+145.0%** | **+126.6%** | **+40.5%** | **+650.0%** |
| | | | | | | | |
| **SVM (RBF)** | *Pre-Tuned Base* | $95.77\%$ | $4.68\%$ | $82.35\%$ | `0.0886` | $0.9426$ | $0.0688$ |
| **SVM (RBF)** | **Final Tuned** | **$96.90\%$** | **$5.74\%$** | **$74.12\%$** | **`0.1065`** | **$0.9354$** | **$0.0728$** |
| *SVM Gain* | *GridSearchCV* | *+1.13%* | **+22.6%** | *-8.23%* | **+20.2%** | *-0.7%* | **+5.8%** |
| | | | | | | | |
| **ANN (MLP)** | *Pre-Tuned Base* | $96.67\%$ | $5.50\%$ | $76.47\%$ | `0.1027` | $0.9105$ | $0.2877$ |
| **ANN (MLP)** | **Final Tuned** | **$97.46\%$** | **$6.95\%$** | **$74.12\%$** | **`0.1271`** | **$0.9090$** | **$0.3047$** |
| *ANN Gain* | *GridSearchCV* | *+0.79%* | **+26.4%** | *-2.35%* | **+23.8%** | *-0.1%* | **+5.9%** |

---

## 🔬 Pipeline Stages (1 to 12)

1. **Stage 1: Ingestion & Integrity Validation** — Ingestion of benign corporate lookups and heavy/light exfiltration PCAP captures.
2. **Stage 2: Schema Normalization & Type Enforcement** — Categorical sanitization and alignment of DNS metadata.
3. **Stage 3: Missing Value Imputation** — Strict zero-leakage median and mode imputation.
4. **Stage 4: Outlier & Robust Scaling** — StandardScaler fitted exclusively on training splits.
5. **Stage 5: Class Imbalance Correction (SMOTE)** — Oversampling the extreme 99.75% / 0.25% class imbalance to a 50/50 balanced training manifold (158,598 samples).
6. **Stage 6: Exploratory Data Analysis** — Correlation heatmaps, entropy distributions, and RFC 1035 packet bounds analysis.
7. **Stage 7: Feature Engineering & Selection** —
   * *Shannon Entropy*: $H(X) = -\sum p(x_i) \log_2 p(x_i)$ distinguishing phonetic domains from Base64 ciphertext.
   * *Interaction Signal*: `entropy_x_length = domain_entropy * query_length` capturing long pseudo-random payloads.
   * *Record Volatility Score*: `tunneling_rr_score = rr_type_TXT + rr_type_NONE - rr_type_A`.
   * *Network Diversity Composite*: Quantifies adversary connections to single C2 nameservers vs. multi-CDN resolutions.
   * *Multicollinearity Reduction*: Dropped 10 zero-variance and 3 collinear features ($|r| \ge 0.95$), leaving 18 features.
8. **Stage 8: Baseline Model (CART Decision Tree)** — Unconstrained benchmark tree diagnosing severe overfitting (Depth: 53, Leaves: 1290, Test F1: 0.1156).
9. **Stage 9: Advanced Model Architectures** — Formulating non-linear kernel SVM (RBF) and Multilayer Perceptron (MLP).
10. **Stage 10: Hyperparameter Tuning (GridSearchCV 5-Fold)** — Stratified cross-validation pruning CART depth to 20 (F1 doubled to 0.2620), optimizing SVM $C=10, \gamma=0.1$, and regularizing ANN topology ($18 \to 64 \to 32 \to 1$).
11. **Stage 11: Comparative Multi-Model Evaluation** — Tri-model ROC curves, Precision-Recall curves, and discrepancy diagnostics.
12. **Stage 12: Production Deployment & Dashboard** — macOS-glassmorphic Flask deployment dashboard with scenario simulations.

---

## 🖥️ Interactive Web Dashboard

The deployment dashboard provides a live interface for security analysts:
* **Predefined Scenario Gallery**: 10 authentic scenarios (5 benign corporate/cloud lookups and 5 attack vectors including Base64 exfil, Base32 credential theft, interactive C2 shell, and stealth A-record tunnels).
* **Tri-Model Simultaneous Prediction**: CART, SVM, and ANN evaluate queries in parallel with animated probability meters.
* **Model Decision Rationale & Geometric Diagnostics**: Explains *why* models agree or disagree based on orthogonal splits, Gaussian distances, and linear dot-product weight saturation.
* **Baseline vs. Tuned Mode Switch**: Live toggle between Pre-Tuned Baseline models and Final Tuned models with dynamic stats updates.
* **Apple macOS Glassmorphism Styling**: Frosted glass panels (`backdrop-filter: blur(24px)`), top-edge specular highlights, and high-contrast orange-black security palette.

---

## ⚡ Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Coder-Harry-J/DNS-tunneling-detector.git
cd DNS-tunneling-detector
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
python run_dashboard.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### 3. Run the Jupyter Notebooks
* `dns_tunneling_stages_1_to_6.ipynb` — Stages 1 to 6 (Preprocessing, Scaling, SMOTE, EDA)
* `dns_tunneling_stages_7_8_10.ipynb` — Stages 7, 8 & 10 (Part 1) (Feature Engineering & CART Tuning)
* `dns_tunneling_stages_9_10_11_12.ipynb` — Stages 9, 10 (Part 2), 11 & 12 (SVM, ANN, Model Comparison & Deployment)

---

## 📂 Repository Structure

```
├── dashboard/
│   ├── app.py                     # Flask REST API backend
│   ├── scenarios_data.json        # 10 authentic scenario vectors & diagnostic rationales
│   └── templates/
│       └── index.html             # macOS-glassmorphic single-page dashboard UI
├── data/
│   └── processed/                 # Preprocessed holdout test set and model artifacts
│       ├── cart_baseline_model.joblib
│       ├── cart_tuned_model.joblib
│       ├── svm_baseline_model.joblib
│       ├── svm_tuned_model.joblib
│       ├── ann_baseline_model.joblib
│       ├── ann_tuned_model.joblib
│       ├── preprocessor_scaler.joblib
│       └── baseline_vs_tuned_full_comparison.csv
├── figures/                       # High-resolution (300 DPI) publication figures
├── dns_tunneling_stages_1_to_6.ipynb
├── dns_tunneling_stages_7_8_10.ipynb
├── dns_tunneling_stages_9_10_11_12.ipynb
├── run_dashboard.py               # Root launcher script
├── requirements.txt
└── README.md
```

---

## 📜 License
MIT License. Created for Network Intrusion Detection & DNS Tunneling Research.
