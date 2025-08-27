from flask import Flask, jsonify
from flask_cors import CORS
from crewai import Crew
from tasks import fetch_weather_task, analyze_weather_task, display_weather_task, Task
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

app = Flask(__name__)
CORS(app)  # Appliquer CORS à toutes les routes

# Helper function to define the tasks with the given hour
def get_tasks_for_hour(hour):
    # Define the analyze task with the specific hour
    analyze_weather_task_hour = Task(
        agent=analyze_weather_task.agent,
        description=analyze_weather_task.description,
        expected_output=analyze_weather_task.expected_output,
        function=lambda: analyze_weather_task.function(hour)
    )

    # Define the display task with the specific hour
    display_weather_task_hour = Task(
        agent=display_weather_task.agent,
        description=display_weather_task.description,
        expected_output=display_weather_task.expected_output,
        function=lambda: display_weather_task.function(hour)
    )

    # Create a Crew instance with the defined tasks
    crew = Crew(tasks=[fetch_weather_task, analyze_weather_task_hour, display_weather_task_hour], verbose=True)
    return crew

# Function to convert CrewOutput to a serializable format (dict or list)
def serialize_crew_output(crew_output):
    if hasattr(crew_output, 'raw_output'):
        result = crew_output.raw_output  # Use raw_output if available
    else:
        result = str(crew_output)  # Fallback: Convert the result to a string

    # If the result is already a dictionary, return it as-is
    if isinstance(result, dict):
        return result

    # If the result is a string, try to parse it as JSON
    if isinstance(result, str):
        import json
        try:
            return json.loads(result)  # Parse the string into a dictionary
        except json.JSONDecodeError:
            return {"result": result}  # Wrap the string in a dictionary if parsing fails

    # If the result is not a dictionary or string, return it as-is
    return result

@app.route("/")  # Home route
def home():
    return "Welcome to the Weather API!"

# Endpoint to get weather for a specific hour
@app.route("/meteo/<int:hour>", methods=["GET"])
def meteo_hour(hour):
    if hour < 0 or hour > 23:
        return jsonify({"error": "Hour must be between 0 and 23"}), 400

    crew = get_tasks_for_hour(hour)
    result = crew.kickoff()  # Execute the tasks for the given hour
    serialized_result = serialize_crew_output(result)  # Convert result to a serializable format
    return jsonify(serialized_result)



# Function to process a single hour in parallel
def process_hour(hour):
    try:
        crew = get_tasks_for_hour(hour)
        result = crew.kickoff()
        return serialize_crew_output(result)
    except Exception as e:
        return {"error": str(e)} 

# Endpoint to get weather for all 24 hours (using parallel processing)
@app.route("/meteo/24_hours", methods=["GET"])
def meteo_24_hours():
    with ThreadPoolExecutor(max_workers=24) as executor:
        # Process all 24 hours in parallel
        results = list(executor.map(process_hour, range(24)))

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)