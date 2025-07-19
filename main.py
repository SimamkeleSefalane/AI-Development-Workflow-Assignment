import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load synthetic dataset
def load_data(filepath='synthetic_patient_data.csv'):
    return pd.read_csv(filepath)
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df):
    df = df.copy()
    
    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])  # Male=1, Female=0

    # Split features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model

# Save the model
def save_model(model, path='readmission_model.joblib'):
    joblib.dump(model, path)

# Main wrapper
def main():
    df = load_data()
    X, y = preprocess(df)
    model = train_model(X, y)
    save_model(model)

if __name__ == "__main__":
    main()
