from crewai import Agent
from weather_api import fetch_weather
import joblib
import numpy as np

# 🔹 Load the trained ML model
model = joblib.load("weather_model.pkl")

# 🔹 ML-based analysis function
def analyze_weather_with_ml(weather_data, hour):
    temp = weather_data.get("temp", 0)
    humidity = weather_data.get("humidity", 0)
    wind_speed = weather_data.get("wind_speed", 0)
    pressure = weather_data.get("pressure", 0)

    X_new = np.array([[temp, humidity, wind_speed, pressure, hour]])
    prediction = model.predict(X_new)[0]

    condition_map = {0: "Clear", 1: "Clouds", 2: "Rain", 3: "Snow", 4: "Drizzle", 5: "Thunderstorm"}
    prediction_text = condition_map.get(prediction, "Unknown")

    # Dynamic advice based on weather conditions
    advice = ""
    if prediction_text == "Clear":
        advice = "It's a clear day! Perfect for outdoor activities like hiking or picnics."
    elif prediction_text == "Clouds":
        advice = "The sky is cloudy, but it's still a good day for a walk or light outdoor activities."
    elif prediction_text == "Rain":
        advice = "It's going to rain. Don't forget your umbrella and consider staying indoors."
    elif prediction_text == "Snow":
        advice = "Snow is expected. Dress warmly and be cautious if driving or walking outside."
    elif prediction_text == "Drizzle":
        advice = "Light drizzle is expected. Carry an umbrella just in case."
    elif prediction_text == "Thunderstorm":
        advice = "Thunderstorms are predicted. Stay indoors and avoid outdoor activities."
    else:
        advice = "Weather conditions are uncertain. Check again later for updates."

    # Return structured data as a dictionary
    return {
        "temperature": f"{temp}°C",
        "humidity": f"{humidity}%",
        "wind_speed": f"{wind_speed} km/h",
        "conditions": prediction_text,
        "advice": advice 
    }


# 🔹 Define the agents
sensor_agent = Agent(
    role="Sensor Agent",
    goal="Retrieve weather data.",
    backstory="An agent that queries a real-time weather API.",
    function=fetch_weather
)

analyzer_agent = Agent(
    role="Analyzer Agent",
    goal="Analyze and predict using an ML model.",
    backstory="An agent that uses a Machine Learning model to predict the weather.",
    function=lambda hour: analyze_weather_with_ml(sensor_agent.function(), hour)
)

visualizer_agent = Agent(
    role="Visualizer Agent",
    goal="Display the weather without modification.",
    backstory="An agent responsible for transmitting weather forecasts to users without modifying them.",
    function=lambda hour: analyzer_agent.function(hour)
)