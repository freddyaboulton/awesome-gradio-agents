from __future__ import annotations as _annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pydantic_core
from gradio.utils import get_upload_folder
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

client = InferenceClient()


@dataclass
class Deps:
    client: InferenceClient


SYSTEM_PROMPT = (
    "You are an assistant designed to help users with their questions. "
    "In addition to having a broad knowledge of many topics you can generate images from a user prompt "
    "with your image generation tool. "
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
async def generate_image(context: RunContext[Deps], prompt: str) -> str:
    """Generate an image pased on a user prompt.
    Returns a local path to the image.

    Args:
        context: The call context.
        prompt: The description for the image to generate.
    """
    img = context.deps.client.text_to_image(
        prompt, model="black-forest-labs/FLUX.1-schnell"
    )
    path = Path(get_upload_folder()) / (uuid.uuid4().hex + ".png")
    img.save(path)
    return str(path)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    yield gr.Textbox(interactive=False, value=""), gr.skip(), gr.skip()
    chatbot.append({"role": "user", "content": prompt})
    yield gr.skip(), chatbot, gr.skip()
    deps = Deps(client=client)
    async with agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    call_args = (
                        call.args.args_dict
                        if hasattr(call.args, "args_dict")
                        else json.dumps(call.args.args_json)
                    )
                    print(call_args)
                    gr_message = {
                        "role": "assistant",
                        "content": f"{call_args}\n",
                        "metadata": {
                            "title": f"🖌️ Generating image",
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
                                f"![Image](/gradio_api/file={call.content})"
                            )
                yield gr.skip(), chatbot, gr.skip()
        chatbot.append({"role": "assistant", "content": ""})
        async for message in result.stream_text():
            chatbot[-1]["content"] = message.replace("sandbox:", "/gradio_api/file=")
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
