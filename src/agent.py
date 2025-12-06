import logging
from typing import Optional

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Configure logging for the agent
logger = logging.getLogger("agent")

# Load environment variables from .env.local file
# This typically contains API keys and configuration settings
load_dotenv(".env.local")


class Assistant(Agent):
    """
    Voice AI Assistant agent that provides helpful responses to user queries.
    
    This assistant is optimized for voice interactions and includes:
    - Concise, natural responses suitable for voice output
    - Weather lookup functionality
    - Friendly and conversational personality
    """
    
    def __init__(self) -> None:
        """Initialize the assistant with system instructions."""
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def get_weather(self, context: RunContext, place: str) -> str:
        """
        Fetch current weather information for a specified location.
        
        Uses the free Open-Meteo API to:
        1. Geocode the location name to coordinates
        2. Retrieve current weather data for those coordinates
        
        Args:
            context: The runtime context for the agent
            place: Name of the location (city, town, etc.)
            
        Returns:
            A formatted string with weather information including:
            - Location name and country
            - Weather description
            - Temperature in Celsius
            - Wind speed in km/h
            
        Raises:
            Exception: If API requests fail or network issues occur
        """
        logger.info("Fetching weather for %s", place)
        
        try:
            # Step 1: Geocode the location name to get coordinates
            async with httpx.AsyncClient(timeout=10) as client:
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={
                        "name": place,
                        "count": 1,  # Only need the top result
                        "language": "en",
                        "format": "json"
                    },
                )
                geo_resp.raise_for_status()
                geo_data = geo_resp.json()

            # Check if any results were found
            results = geo_data.get("results") or []
            if not results:
                return (
                    f"Couldn't find weather for '{place}'. "
                    "Try a nearby city or add a country."
                )

            # Extract location data from the first result
            loc = results[0]
            latitude = loc["latitude"]
            longitude = loc["longitude"]
            loc_name = loc.get("name", place)
            country = loc.get("country", "")
            
            # Step 2: Fetch weather data using the coordinates
            async with httpx.AsyncClient(timeout=10) as client:
                weather_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "current_weather": "true",
                        "timezone": "auto",
                    },
                )
                weather_resp.raise_for_status()
                weather_data = weather_resp.json()
                
            # Extract current weather information
            current = weather_data.get("current_weather")
            if not current:
                return (
                    f"Weather unavailable for {loc_name} {country}".strip()
                    or "Weather unavailable."
                )

            # WMO Weather interpretation codes
            # Maps numeric codes to human-readable descriptions
            code_map = {
                0: "clear sky",
                1: "mainly clear",
                2: "partly cloudy",
                3: "overcast",
                45: "fog",
                48: "depositing rime fog",
                51: "light drizzle",
                53: "moderate drizzle",
                55: "dense drizzle",
                56: "freezing drizzle",
                57: "freezing drizzle",
                61: "light rain",
                63: "moderate rain",
                65: "heavy rain",
                66: "freezing rain",
                67: "freezing rain",
                71: "light snow",
                73: "moderate snow",
                75: "heavy snow",
                77: "snow grains",
                80: "light showers",
                81: "moderate showers",
                82: "violent showers",
                85: "light snow showers",
                86: "heavy snow showers",
                95: "thunderstorm",
                96: "thunderstorm with slight hail",
                99: "thunderstorm with heavy hail",
            }

            # Parse weather data
            description = code_map.get(current.get("weathercode"), "unknown")
            temperature = current.get("temperature")
            windspeed = current.get("windspeed")

            # Format and return the weather report
            location_label = f"{loc_name}, {country}".strip(", ")
            return (
                f"{location_label}: {description}, "
                f"{temperature}°C, wind {windspeed} km/h"
            )
            
        except httpx.HTTPError as e:
            # Handle HTTP-specific errors
            logger.error("HTTP error while fetching weather: %s", e)
            return f"Unable to fetch weather data for {place}. Please try again later."
        except Exception as e:
            # Handle any other unexpected errors
            logger.error("Unexpected error fetching weather: %s", e)
            return f"An error occurred while getting weather information for {place}."


# Create the agent server instance
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """
    Prewarm function to load models before handling sessions.
    
    This function runs during the agent startup phase to:
    - Load the Voice Activity Detection (VAD) model into memory
    - Store it in process userdata for reuse across sessions
    - Reduce latency when starting new sessions
    
    Args:
        proc: The job process being prewarmed
    """
    logger.info("Prewarming: Loading VAD model")
    proc.userdata["vad"] = silero.VAD.load()


# Set the prewarm function to be called during startup
server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    """
    Main agent session handler for real-time voice conversations.
    
    This function:
    1. Sets up logging context for the session
    2. Configures the agent with STT, LLM, TTS, and VAD models
    3. Initializes the session with the Assistant agent
    4. Handles audio input with noise cancellation
    5. Connects to the LiveKit room
    
    Args:
        ctx: Job context containing room and process information
    """
    # Add context fields for better log tracking
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create an agent session with all necessary components
    session = AgentSession(
        # Speech-to-Text: Transcribes user voice input
        stt=inference.STT(
            model="assemblyai/universal-streaming",
            language="en"
        ),
        
        # Large Language Model: Generates responses
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        
        # Text-to-Speech: Converts agent responses to voice
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # Specific voice ID
        ),
        
        # Detects when user has finished speaking
        turn_detection=MultilingualModel(),
        
        # Voice Activity Detection: Distinguishes speech from silence
        vad=ctx.proc.userdata["vad"],
        
        # Generate responses preemptively for lower latency
        preemptive_generation=True,
    )

    # Start the session with the assistant agent
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Apply different noise cancellation based on participant type
                # SIP participants use telephony-optimized noise cancellation
                # Regular participants use standard noise cancellation
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Connect to the LiveKit room and start handling events
    await ctx.connect()


if __name__ == "__main__":
    # Run the agent server using the LiveKit CLI
    cli.run_app(server)