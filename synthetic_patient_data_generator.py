import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000, random_state=42):
    np.random.seed(random_state)

    data = {
        'age': np.random.randint(18, 90, size=num_samples),
        'gender': np.random.choice(['Male', 'Female'], size=num_samples),
        'blood_pressure': np.random.normal(120, 15, size=num_samples),
        'cholesterol': np.random.normal(200, 30, size=num_samples),
        'diabetes': np.random.choice([0, 1], size=num_samples),
        'num_previous_admissions': np.random.poisson(1, size=num_samples),
        'readmitted': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Introduce some missing values
    for col in ['blood_pressure', 'cholesterol']:
        df.loc[df.sample(frac=0.1, random_state=random_state).index, col] = np.nan

    return df

def save_data(df, filename='synthetic_patient_data.csv'):
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

if __name__ == "__main__":
    df = generate_synthetic_data()
    save_data(df)
