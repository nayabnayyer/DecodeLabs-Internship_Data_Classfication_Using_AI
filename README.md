# 🛡️ FraudGuard AI Enterprise

A premium, institutional-grade fraud detection dashboard built with Python and Streamlit — combining a SMOTE-balanced Random Forest classifier with a real-time risk intelligence interface styled after enterprise fintech platforms.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-orange?style=flat-square&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ What Is This?

FraudGuard isn't just a model with a UI bolted on — it's a full **risk operations dashboard**, the kind a bank's fraud team might actually use. Upload a transaction ledger, the system trains a SMOTE-balanced Random Forest live, then gives you a multi-pane control center to monitor, investigate, and act on fraud signals in real time.

---

## 🚀 Features

| Module | Description |
|---|---|
| 📊 **Risk Intelligence Suite** | Live KPIs, volumetric risk scatter plot, real-time incident stream, single-transaction forensic scoring |
| 💳 **Transaction Ledger** | Direct view into the uploaded dataset |
| 🚩 **Escalations Management** | Auto-filtered queue of every transaction flagged as fraud this session |
| 📈 **Hyperparameter Validation** | Confusion matrix heatmap + cross-model benchmark comparison (Logistic Regression, Decision Tree, KNN, Random Forest) |
| ⚙️ **System Preferences** | Adjustable sensitivity threshold and logging toggles |

---

## 🧠 The Model

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Class imbalance fix:** SMOTE oversampling on the training set only — prevents the classic "99% accuracy, 0% recall" trap common in fraud datasets
- **Scaling:** StandardScaler applied post-SMOTE
- **Evaluation:** Accuracy, F1, ROC-AUC, full confusion matrix, and classification report — all computed live and cached with `@st.cache_resource` so retraining only happens once per uploaded file

```python
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

This single line is what separates a real fraud detector from a model that just learns to say "not fraud" every time.

---

## 🎨 Design

Styled as a **premium Swiss banking interface** — ink navy backgrounds, mint-green compliance accents, danger-red escalation states, and glassmorphic header blur. Every panel, metric, and alert uses a consistent card system (`fg-card`) for a cohesive enterprise look rather than default Streamlit styling.

Color system:
- `#00E676` — Mint compliance green (cleared transactions)
- `#FF5252` — Risk red (fraud escalations)
- `#FFD740` — Amber (mid-risk warning band)

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- A labeled transaction CSV with a `Class` column (0 = normal, 1 = fraud) — e.g. the popular Kaggle Credit Card Fraud dataset

### Installation
```bash
git clone https://github.com/your-username/fraudguard-ai.git
cd fraudguard-ai
pip install streamlit pandas numpy scikit-learn imbalanced-learn plotly joblib
```

### Run
```bash
streamlit run fraudguard.py
```

Open **http://localhost:8501**, upload your CSV in the **System Ingestion Pipeline** section, and the model trains automatically.

---

## 🕹️ How to Use

1. **Upload** a transaction CSV with `Time`, `V1`–`V28`, `Amount`, and `Class` columns
2. The **Risk Intelligence Suite** activates — view live metrics and the risk scatter plot
3. Use **Single Instance Forensic Entry** to manually score a transaction, or click **Inject Anomaly Profile** / **Inject Compliant Profile** to auto-fill a real example from the data
4. Click **Execute Anomaly Check** to see the live probability gauge and clearance/rejection banner
5. Check **Escalations Management** to review every transaction flagged as fraud this session
6. Visit **Hyperparameter Validation** to compare your model against benchmark algorithms

---

## 📁 Project Structure

```
DecodeLabs-Internship_Data_Classification_Using_AI
│
├── Fraud_Detection(4).py     # Complete application — one file
└── README.md         # You're reading this
```

---

## 🛣️ Roadmap

- Persist trained models with `joblib` between sessions
- Add user authentication for multi-analyst access
- Export escalation queue to CSV/PDF report
- Add SHAP-based explainability for individual predictions
- Support live streaming transaction feeds (Kafka/WebSocket)

---

## 👩‍💻 Author

**Nayab Nayyer**
Fresh CS Graduate · Python · Machine Learning · Streamlit
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/nayab-nayyer-2b6803321)

---

## 📄 License

Open source under the [MIT License](LICENSE).
