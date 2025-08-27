import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE
from weather_api import fetch_weather

# 🔹 Encode weather condition into numerical categories
def encode_weather_condition(df):
    condition_map = {"Clear": 0, "Clouds": 1, "Rain": 2, "Snow": 3, "Drizzle": 4, "Thunderstorm": 5}
    df["condition"] = df["condition"].map(condition_map).fillna(1)  # Default to "Clouds"
    return df

# 🔹 Train the model
def train_model():
    df = fetch_weather()
    if df is None:
        print("❌ Unable to fetch weather data.")
        return

    df = encode_weather_condition(df)

    # Features (température, humidité, vent, pression, et l'heure)
    X = df[["temp", "humidity", "wind_speed", "pressure", "hour"]]  # Ajouter l'heure comme feature
    y = df["condition"]

    # Normaliser les données
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer SMOTE si nécessaire
    class_counts = y.value_counts()
    if class_counts.min() > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(2, class_counts.min() - 1))
        X_res, y_res = smote.fit_resample(X_scaled, y)
    else:
        print("⚠️ Not enough samples in some classes. Skipping SMOTE.")
        X_res, y_res = X_scaled, y

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Optimisation des hyperparamètres avec GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5), scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Meilleur modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Optimized model accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Sauvegarder le modèle et le scaler
    joblib.dump(best_model, "weather_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model and scaler saved!")

if __name__ == "__main__":
    train_model()