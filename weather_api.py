import requests
import pandas as pd

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Casablanca"
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# 🔹 Modifier la fonction pour obtenir les prévisions horaires
def fetch_weather():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        weather_data = []
        for entry in data["list"]:
            weather_data.append({
                "temp": entry["main"]["temp"],
                "humidity": entry["main"]["humidity"],
                "wind_speed": entry["wind"]["speed"],
                "pressure": entry["main"]["pressure"],
                "condition": entry["weather"][0]["main"],
                "dt": entry["dt"],  # Timestamp pour l'heure
            })
        df = pd.DataFrame(weather_data)
        df["hour"] = pd.to_datetime(df["dt"], unit="s").dt.hour  # Extraire l'heure de la timestamp
        return df
    return None
