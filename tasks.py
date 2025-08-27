from crewai import Task
from agents import sensor_agent, analyzer_agent, visualizer_agent

# 🔹 Fetcher Task: Fetch weather data
fetch_weather_task = Task(
    agent=sensor_agent,
    description="Fetch current weather data from OpenWeatherMap.",
    expected_output="A dictionary containing temperature, humidity, wind speed, and a description of weather conditions."
)

# 🔹 Analyzer Task: Analyze and predict trends
analyze_weather_task = Task(
    agent=analyzer_agent,
    description="Analyze the received weather data and predict future conditions.",
    expected_output="A dictionary containing exactly the following keys and their values : temperature (float number), humidity (in percentage), wind speed(float number), weather conditions, and advice."
)

# 🔹 Visualizer Task: Display forecasts
display_weather_task = Task(
    agent=visualizer_agent,
    description="Display weather forecasts to users without modifying them.",
    expected_output="A dictionary identical to the one provided by the analyzer agent, without any modifications."
)