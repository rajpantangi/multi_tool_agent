import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent # type: ignore
from google.adk.tools import google_search

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


# A specialized sub-agent for handling weather and time tools.
weather_time_sub_agent = Agent(
    name="weather_time_sub_agent",
    model="gemini-1.5-flash-001",
    description=(
        "A specialized agent that can get the current weather and time for a"
        " specific city."
    ),
    instruction=(
        "You are a helpful agent that answers user questions about the time and"
        " weather in a city using your tools."
    ),
    tools=[get_weather, get_current_time],
)

# The root agent acts as a coordinator. It uses Google Search for general
# questions and delegates weather/time questions to its sub-agent.
root_agent = Agent(
    name="coordinator_agent",
    model="gemini-1.5-flash-001",
    description=(
        "A coordinator agent that can perform Google searches for general"
        " information and delegate tasks about weather and time to a specialized"
        " sub-agent."
    ),
    instruction=(
        "You are a helpful coordinator agent. Your primary role is to route user"
        " requests. If the user asks a general knowledge question, use the"
        " google_search tool. If the user asks about the weather or time in a"
        " city, you MUST delegate the task to the `weather_time_sub_agent`."
    ),
    tools=[google_search],
    sub_agents=[weather_time_sub_agent],
)