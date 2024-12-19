import json
import os
from typing import Any

import gradio as gr
import requests
from transformers import ReactCodeAgent, Tool, tool
from transformers.agents import HfApiEngine, stream_to_gradio


@tool
def get_lat_lng(location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        location_description: A description of a location.
    """
    if os.getenv("GEO_API_KEY") is None:
        # if no API key is provided, return a dummy response (London)
        return {"lat": 51.1, "lng": -0.1}

    params = {
        "q": location_description,
        "api_key": os.getenv("GEO_API_KEY"),
    }

    r = requests.get("https://geocode.maps.co/search", params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        return {"lat": data[0]["lat"], "lng": data[0]["lon"]}
    else:
        raise ValueError("Could not find the location")


@tool
def get_weather(lat: float, lng: float) -> str:
    """Get the weather at a location.

    Args:
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if not os.getenv("WEATHER_API_KEY"):
        # if no API key is provided, return a dummy response
        return json.dumps({"temperature": "21 °C", "description": "Sunny"})

    params = {
        "apikey": os.getenv("WEATHER_API_KEY"),
        "location": f"{lat},{lng}",
        "units": "metric",
    }

    r = requests.get("https://api.tomorrow.io/v4/weather/realtime", params=params)
    r.raise_for_status()
    data = r.json()

    values = data["data"]["values"]
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm",
    }
    return json.dumps(
        {
            "temperature": f'{values["temperatureApparent"]:0.0f}°C',
            "description": code_lookup.get(values["weatherCode"], "Unknown"),
        }
    )


llm_engine = HfApiEngine("Qwen/Qwen2.5-Coder-32B-Instruct")
# Initialize the agent with both tools and engine
agent = ReactCodeAgent(tools=[get_weather, get_lat_lng], llm_engine=llm_engine)


def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append(gr.ChatMessage(role="user", content=prompt))
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip()
    for msg in stream_to_gradio(agent, prompt):
        chatbot.append(msg)
        yield gr.skip(), chatbot, gr.skip()
    yield gr.Textbox(interactive=True), chatbot, gr.skip()


def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    yield from stream_from_agent(previous_prompt, new_history, past_messages)


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value["text"]
