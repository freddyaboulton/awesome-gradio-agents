import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
import gradio as gr
import pydantic_core
from openai import AsyncOpenAI
from pydantic import BaseModel
from transformers import ReactJsonAgent, tool
from transformers.agents import HfApiEngine, stream_to_gradio

current_dir = Path(__file__).parent
json_file = str(current_dir / "gradio_docs.json")

DOCS = json.load(open(json_file, "r"))

openai = AsyncOpenAI()


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


class RetrievalResult(BaseModel):
    content: str
    ids: list[int]


async def _retrieve(search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    print(f"create embedding for {search_query}")
    embedding = await openai.embeddings.create(
        input=search_query,
        model="text-embedding-3-small",
    )

    assert (
        len(embedding.data) == 1
    ), f"Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}"
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()

    async with database_connect(False) as pool:
        rows = await pool.fetch(
            "SELECT id, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8",
            embedding_json,
        )
        content = "\n\n".join(f'# {row["title"]}\n{row["content"]}\n' for row in rows)
        ids = [row["id"] for row in rows]
        return RetrievalResult(content=content, ids=ids).model_dump_json()


@tool
def retrieve(search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    return asyncio.run(_retrieve(search_query))


llm_engine = HfApiEngine("Qwen/Qwen2.5-Coder-32B-Instruct")
# Initialize the agent with both tools and engine
agent = ReactJsonAgent(tools=[retrieve], llm_engine=llm_engine)


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
