import gradio as gr
from transformers import ReactCodeAgent, Tool  # type: ignore
from transformers.agents import HfApiEngine, stream_to_gradio  # type: ignore

# Import tool from Hub
image_generation_tool = Tool.from_space(
    space_id="black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generates an image following your prompt. Returns a PIL Image.",
    api_name="/infer",
)

llm_engine = HfApiEngine("Qwen/Qwen2.5-Coder-32B-Instruct")
# Initialize the agent with both tools and engine
agent = ReactCodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)


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
