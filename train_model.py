import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train():
    print("Loading data...")
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        print("Error: heart.csv not found.")
        return

    # 1. Clean Data
    print("Cleaning data...")
    # Drop useless columns if present
    for col in ['id', 'dataset']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Handle Missing Values
    # Numeric -> Median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical -> Mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 2. Prepare Features & Target
    # Convert target 'num' to Binary (0 vs 1+)
    # 0 = No Disease, 1-4 = Disease
    if 'num' in df.columns:
        y = (df['num'] > 0).astype(int)
        X = df.drop(columns=['num'])
    else:
        print("Error: 'num' column not found.")
        return

    # 3. Encoding
    print("Encoding categorical features...")
    encoders = {}
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Save Encoders
    joblib.dump(encoders, 'encoders.pkl')
    print("Saved encoders.pkl")

    # 4. Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved scaler.pkl")

    # 6. Train Model
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 7. Save Model
    joblib.dump(model, 'heart_model.pkl')
    print("Saved heart_model.pkl")

if __name__ == "__main__":
    train()
