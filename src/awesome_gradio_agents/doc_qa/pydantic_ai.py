from __future__ import annotations as _annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
import gradio as gr
import pydantic_core
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

current_dir = Path(__file__).parent
json_file = str(current_dir / "gradio_docs.json")

DOCS = json.load(open(json_file, "r"))

openai = AsyncOpenAI()


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


SYSTEM_PROMPT = (
    "You are an assistant designed to help users answer questions about Gradio. "
    "You have a retrieve tool that can provide relevant documentation sections based on the user query. "
    "Be curteous and helpful to the user but feel free to refuse answering questions that are not about Gradio. "
)


agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    system_prompt=SYSTEM_PROMPT,
)


class RetrievalResult(BaseModel):
    content: str
    ids: list[int]


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        os.getenv("DB_URL"),
        "gradio_ai_rag",
    )
    if create_db:
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
            if not db_exists:
                await conn.execute(f"CREATE DATABASE {database}")
        finally:
            await conn.close()

    pool = await asyncpg.create_pool(f"{server_dsn}/{database}")
    try:
        yield pool
    finally:
        await pool.close()


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    print(f"create embedding for {search_query}")
    embedding = await context.deps.openai.embeddings.create(
        input=search_query,
        model="text-embedding-3-small",
    )

    assert (
        len(embedding.data) == 1
    ), f"Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}"
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        "SELECT id, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8",
        embedding_json,
    )
    content = "\n\n".join(f'# {row["title"]}\n{row["content"]}\n' for row in rows)
    ids = [row["id"] for row in rows]
    return RetrievalResult(content=content, ids=ids).model_dump_json()


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    yield gr.Textbox(interactive=False, value=""), gr.skip(), gr.skip()
    chatbot.append({"role": "user", "content": prompt})
    yield gr.skip(), chatbot, gr.skip()

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        async with agent.run_stream(
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
                        print(call_args)
                        gr_message = {
                            "role": "assistant",
                            "content": "",
                            "metadata": {
                                "title": f"🔍 Retrieving Relevant Docs",
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
                                paths = []
                                for d in DOCS:
                                    tool_result = RetrievalResult.model_validate_json(
                                        call.content
                                    )
                                    if d["id"] in tool_result.ids:
                                        paths.append(d["path"])
                                paths = "\n".join(list(set(paths)))
                                gr_message["content"] = f"Relevant Context:\n {paths}"
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
