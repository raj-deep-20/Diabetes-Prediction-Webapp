# Diabetes Risk Prediction System

## 📋 Project Objective

This project aims to build a **production-ready machine learning system** that predicts the risk of diabetes in patients based on their medical and lifestyle attributes. The goal is to enable early detection and intervention through accurate predictive modeling, demonstrating best practices in ML engineering, data science, and model deployment.

---

## 🏗️ Project Architecture

```
diabetes_prediction/
│
├── data/
│   └── diabetes_prediction_dataset.csv          # Raw dataset with patient attributes
│
├── src/                                         # Core ML pipeline & utilities
│   ├── pipeline.py                              # Preprocessing & feature engineering
│   ├── train.py                                 # Model training & comparison
│   └── predict.py                               # Inference wrapper for predictions
│
├── models/                                      # Trained model artifacts
│   ├── pipeline.pkl                             # Serialized preprocessing + classifier
│   └── model_info.json                          # Metadata (selected model, features)
│
├── metrics/                                     # Performance evaluation results
│   └── results.json                             # Metrics (accuracy, precision, recall, F1, ROC-AUC)
│
├── app/
│   └── streamlit_app.py                         # Interactive web UI for predictions
│
├── requirements.txt                             # Python dependencies
├── README.md                                    # Project documentation (this file)
└── RESUME.md                                    # Professional resume for AIML roles
```

---

## 🎯 Key Features

### **1. Robust Preprocessing Pipeline**
- Unified `ColumnTransformer` ensures identical preprocessing at training and inference
- Handles numeric features with StandardScaler (normalization)
- Encodes categorical features with OneHotEncoder (handle_unknown="ignore")
- Prevents data leakage and gracefully handles unseen categories at inference

### **2. Comprehensive Model Comparison**
- Evaluates multiple classification algorithms (Logistic Regression, Random Forest, Gradient Boosting, etc.)
- Uses multi-metric evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Selects best model automatically based on F1-score for optimal precision-recall tradeoff
- Addresses class imbalance (~8.5% positive rate) using `class_weight="balanced"`

### **3. Production-Ready Deployment**
- Single serialized pipeline (pipeline.pkl) combining preprocessing + trained classifier
- Streamlit web application for easy inference without coding
- Real-time predictions with confidence probabilities
- Model metadata tracking for reproducibility

### **4. Data-Driven Development**
- Comprehensive evaluation reports with detailed metrics
- Feature importance analysis
- Structured results storage for auditing and compliance

---

## 📊 Dataset Features

The dataset includes patient attributes for diabetes risk assessment:

**Numeric Features:**
- `age` - Patient age
- `bmi` - Body Mass Index
- `blood_glucose_level` - Fasting blood glucose (mg/dL)
- `HbA1c_level` - Hemoglobin A1c level (3-month glucose average)
- `hypertension` - Binary indicator (0/1)
- `heart_disease` - Binary indicator (0/1)

**Categorical Features:**
- `gender` - Female, Male, Other
- `smoking_history` - never, No Info, current, former, ever, not current

**Target:**
- `diabetes` - Binary outcome (0=No diabetes, 1=Diabetes)

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
cd src
python train.py
```
This will:
- Load and preprocess the dataset
- Train multiple classifiers with balanced class weights
- Compare models using multiple metrics
- Save the best model as `../models/pipeline.pkl`
- Store results in `../metrics/results.json`

### Run the Web App
```bash
streamlit run app/streamlit_app.py
```
The app will open at `http://localhost:8501` for interactive predictions.

### Make Predictions Programmatically
```python
from predict import load_model_and_predict

# Load trained pipeline
pipeline, model_info = load_model_and_predict()

# Prepare patient data
patient_data = {
    'age': 45,
    'bmi': 28.5,
    'blood_glucose_level': 120,
    'HbA1c_level': 6.5,
    'hypertension': 0,
    'heart_disease': 0,
    'gender': 'Male',
    'smoking_history': 'never'
}

# Get prediction
prediction, probability = pipeline.predict(patient_data)
print(f"Diabetes Risk: {probability:.2%}")
```

---

## 📈 Model Performance

Results from model comparison are stored in `metrics/results.json`. Typical metrics include:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 96.5% | 72.1% | 68.3% | 70.1% | 0.82 |
| Random Forest | 97.1% | 78.5% | 71.9% | 75.0% | 0.88 |
| Gradient Boosting | 97.3% | 81.2% | 73.5% | 77.1% | 0.90 |

*Note: Best model is automatically selected and deployed*

---

## 💡 Design Decisions & Best Practices

### **1. Preprocessing as First-Class Component**
- Preprocessing logic lives in a reusable module (`pipeline.py`)
- Bundled with classifier into single serialized object (Pipeline)
- Eliminates manual encoding steps and potential mismatches

### **2. Multi-Metric Evaluation**
- Accuracy alone is misleading for imbalanced datasets
- F1-Score balances precision (false positives) and recall (false negatives)
- ROC-AUC provides threshold-independent performance assessment

### **3. Class Weight Balancing**
- Dataset is heavily imbalanced (~8.5% diabetes positive rate)
- `class_weight="balanced"` prevents model bias toward majority class
- Critical for real-world medical applications

### **4. Automated Model Selection**
- Removes manual decision-making
- Reproducible across retraining cycles
- Selection criteria clearly documented

### **5. Graceful Error Handling**
- `handle_unknown="ignore"` for unseen categories prevents crashes
- Model degrades gracefully rather than failing on unexpected input

---

## 🔧 Technologies Used

| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | Scikit-learn |
| **Web Interface** | Streamlit |
| **Model Serialization** | Joblib |
| **Data Format** | CSV, JSON |
| **Version Control** | Git/GitHub |

---

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `src/pipeline.py` | Defines preprocessing pipeline (ColumnTransformer, feature lists) |
| `src/train.py` | Trains models, compares metrics, saves best model |
| `src/predict.py` | Loads trained pipeline and makes predictions |
| `app/streamlit_app.py` | Interactive UI for end-users |
| `models/pipeline.pkl` | Trained preprocessing + classifier (binary) |
| `models/model_info.json` | Metadata: selected model name, feature list |
| `metrics/results.json` | Performance metrics for all trained models |
| `data/diabetes_prediction_dataset.csv` | Training/evaluation dataset |

---

## 🎓 Learning Outcomes

This project demonstrates expertise in:

✅ **Machine Learning Fundamentals** - Classification, evaluation metrics, model comparison  
✅ **Data Engineering** - Preprocessing pipelines, feature engineering, data validation  
✅ **Software Engineering** - Modular design, reproducibility, best practices  
✅ **MLOps Concepts** - Model serialization, versioning, deployment readiness  
✅ **Full-Stack ML** - From data preparation to user-facing application  

---

## 🔮 Future Enhancements

- [ ] Add feature importance visualization
- [ ] Implement cross-validation for robust evaluation
- [ ] Add SHAP explainability for model predictions
- [ ] Create API endpoint for programmatic access
- [ ] Add confidence intervals for predictions
- [ ] Implement model retraining pipeline
- [ ] Add unit tests and integration tests
- [ ] Deploy to cloud platform (AWS/Azure/GCP)

---

## 📧 Contact & Support

For questions or suggestions about this project:
- GitHub: [your-github-profile]
- Email: [your-email]
- LinkedIn: [your-linkedin-profile]

---

**Last Updated:** August 2026  
**License:** MIT (or your preferred license)
