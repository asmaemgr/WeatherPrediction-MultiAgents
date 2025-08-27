from crewai import Crew
from tasks import fetch_weather_task, analyze_weather_task, display_weather_task, Task
import os
import json  # Import the json module
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Get the hour from the user
hour = int(input("Enter the hour for which you want a forecast (0-23): "))

# Define tasks with the hour
analyze_weather_task_hour = Task(
    agent=analyze_weather_task.agent,
    description=analyze_weather_task.description,
    expected_output=analyze_weather_task.expected_output,
    function=lambda: analyze_weather_task.function(hour)
)

display_weather_task_hour = Task(
    agent=display_weather_task.agent,
    description=display_weather_task.description,
    expected_output=display_weather_task.expected_output,
    function=lambda: display_weather_task.function(hour) 
)

# Define Crew with the tasks
crew = Crew(tasks=[fetch_weather_task, analyze_weather_task_hour, display_weather_task_hour], verbose=True)

if __name__ == "__main__":
    result = crew.kickoff()
    print(type(result))  # Debugging: Check the type of the result

    # Extract the relevant data from the CrewOutput object
    if hasattr(result, 'raw_output'):  # Check if the result has a 'raw_output' attribute
        weather_data = result.raw_output  # Assuming the raw_output contains the weather data
    else:
        weather_data = str(result)  # Fallback: Convert the result to a string

    # Convert the weather data to a JSON-compatible dictionary
    if isinstance(weather_data, dict):  # If it's already a dictionary
        json_output = weather_data
    else:
        # If it's a string, parse it into a dictionary (if possible)
        try:
            json_output = json.loads(weather_data)
        except json.JSONDecodeError:
            # If parsing fails, wrap the result in a dictionary
            json_output = {"result": weather_data}

    # Print the JSON output with special characters preserved
    print("\n🌦️ Final Weather Forecast:\n", json.dumps(json_output, indent=4, ensure_ascii=False))