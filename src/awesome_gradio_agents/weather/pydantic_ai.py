from __future__ import annotations as _annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import gradio as gr
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ToolCallPart, ToolReturnPart


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent(
    "openai:gpt-4o",
    system_prompt="Be concise, reply with one sentence.",
    deps_type=Deps,
    retries=2,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {"lat": 51.1, "lng": -0.1}

    params = {
        "q": location_description,
        "api_key": ctx.deps.geo_api_key,
    }

    r = await ctx.deps.client.get("https://geocode.maps.co/search", params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        return {"lat": data[0]["lat"], "lng": data[0]["lon"]}
    else:
        raise ModelRetry("Could not find the location")


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {"temperature": "21 °C", "description": "Sunny"}

    params = {
        "apikey": ctx.deps.weather_api_key,
        "location": f"{lat},{lng}",
        "units": "metric",
    }

    r = await ctx.deps.client.get(
        "https://api.tomorrow.io/v4/weather/realtime", params=params
    )
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
    return {
        "temperature": f'{values["temperatureApparent"]:0.0f}°C',
        "description": code_lookup.get(values["weatherCode"], "Unknown"),
    }


TOOL_TO_DISPLAY_NAME = {"get_lat_lng": "Geocoding API", "get_weather": "Weather API"}

client = AsyncClient()
weather_api_key = os.getenv("WEATHER_API_KEY")
# create a free API key at https://geocode.maps.co/
geo_api_key = os.getenv("GEO_API_KEY")
deps = Deps(client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append({"role": "user", "content": prompt})
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip()
    async with weather_agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    call_args = (
                        call.args.args_json
                        if hasattr(call.args, "args_json")
                        else json.dumps(call.args.args_dict)
                    )
                    gr_message = {
                        "role": "assistant",
                        "content": "Parameters: " + call_args,
                        "metadata": {
                            "title": f"🛠️ Using {TOOL_TO_DISPLAY_NAME[call.tool_name]}",
                            "id": call.tool_call_id,
                        },
                    }
                    chatbot.append(gr_message)
                if isinstance(call, ToolReturnPart):
                    for gr_message in chatbot:
                        if (
                            gr_message.get("metadata", {}).get("id", "")
                            == call.tool_call_id
                        ):
                            gr_message["content"] += (
                                f"\nOutput: {json.dumps(call.content)}"
                            )
                yield gr.skip(), chatbot, gr.skip()
        chatbot.append({"role": "assistant", "content": ""})
        async for message in result.stream_text():
            chatbot[-1]["content"] = message
            yield gr.skip(), chatbot, gr.skip()
        past_messages = result.all_messages()

        yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value["text"]
