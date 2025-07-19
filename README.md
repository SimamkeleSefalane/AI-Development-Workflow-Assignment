# AI-Development-Workflow-Assignment

# 🏥 Hospital Readmission Risk Prediction

This repository implements a simulated end-to-end AI workflow to predict whether a patient will be readmitted to a hospital within 30 days of discharge.

## 📂 Contents

- `readmission_prediction_notebook.ipynb`: Jupyter notebook containing the full pipeline.
- `synthetic_patient_data_generator.py`: Script to generate synthetic data.
- `readmission_prediction.py`: Cleaned Python script version of the notebook.
- `synthetic_patient_data.csv`: Simulated dataset.
- `requirements.txt`: List of required Python libraries.

## 🚀 Project Workflow

1. **Problem Definition**: Predict patient readmission within 30 days to improve care.
2. **Data Generation**: Create synthetic patient health records with missing values.
3. **Preprocessing**: Impute missing data, scale features.
4. **Model Development**: Train a Logistic Regression model.
5. **Evaluation**: Assess performance using confusion matrix, precision, and recall.
6. **(Optional) Deployment**: Save the model or integrate it in a Flask/Streamlit app.

## 📊 Example Metrics
- **Precision**: 77.8%
- **Recall**: 70.0%
- **Model**: Logistic Regression

## ⚠️ Ethical Considerations
- Data privacy (use synthetic data only).
- Potential bias in training datasets.
- Transparent and interpretable models are preferred in healthcare.

## 🧪 How to Run
```bash
pip install -r requirements.txt
python synthetic_patient_data_generator.py
python readmission_prediction.py
```
Or open the Jupyter notebook:
```bash
jupyter notebook readmission_prediction_notebook.ipynb
```

## 📝 License
MIT License

## 🤝 Contributing
This project was completed as part of the PLP Academy AI Workflow Assignment by Simamkele Sefalane and team.
