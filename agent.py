import datetime
from zoneinfo import ZoneInfo
import subprocess
import json
from google.adk.agents import Agent # type: ignore

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_current_location() -> dict:
    """Retrieves the current location (city, region, country) using an external IP lookup service.

    Returns:
        dict: A dictionary containing the status and location information, or an error message.
    """
    try:
        # Execute the curl command to get the IP information
        result = subprocess.run(['curl', 'https://ipwho.is/'], capture_output=True, text=True, check=True)
        
        # Parse the JSON output
        data = json.loads(result.stdout)

        # Extract relevant information
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')

        location_string = f"{city}, {region}, {country}"
        return {"status": "success", "location": location_string}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "error_message": f"Error executing curl: {e}"}
    except json.JSONDecodeError:
        return {"status": "error", "error_message": "Failed to decode JSON from the IP lookup service."}
    except Exception as e:
        return {"status": "error", "error_message": f"An unexpected error occurred: {e}"}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer general questions and about the time and weather in a city."
    ),
    instruction=(
        "Answer questions using Google Search when needed. Always cite sources. You can use the tools provided for answering any question about weather and time of a particular city." \
        "If the user asks about his current location use the get_current_location tool to get the current location and return it"
    ),
    tools=[get_weather, get_current_time, get_current_location],
    sub_agents=[]
)
