import gradio as gr
from awesome_gradio_agents.utils import Logo, load_dotenv


def make_app(fn, undo_fn, retry_fn, select_fn, library: str):
    with gr.Blocks() as demo:
        gr.HTML(f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; padding: 1rem; width: 100%">
        <img src={Logo[library].value} style="max-width: 200px; height: auto">
        <div>
            <h1 style="margin: 0 0 1rem 0">Chat with Gradio Docs 🗣️</h1>
            <h3 style="margin: 0 0 0.5rem 0">
                Simple RAG agent over Gradio docs.
            </h3>
        </div>
    </div>
        """)
        past_messages = gr.State([])
        chatbot = gr.Chatbot(
            label="Gradio Docs Bot",
            type="messages",
            examples=[
                {"text": "How to display images in a Gradio chatbot?"},
                {"text": "How to stream audio in Gradio?"},
            ],
            avatar_images=(None, Logo[library].value),
        )
        with gr.Row():
            prompt = gr.Textbox(
                lines=1,
                show_label=False,
                placeholder="How to build a calculator with Gradio?",
            )
        generation = prompt.submit(
            fn,
            inputs=[prompt, chatbot, past_messages],
            outputs=[prompt, chatbot, past_messages],
        )
        chatbot.example_select(select_fn, None, [prompt])
        chatbot.retry(
            retry_fn, [chatbot, past_messages], [prompt, chatbot, past_messages]
        )
        chatbot.undo(
            undo_fn, [chatbot, past_messages], [prompt, chatbot, past_messages]
        )

        return demo


if __name__ == "__main__":
    import argparse

    # parse the library argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, default="pydantic_ai")
    args = parser.parse_args()
    load_dotenv()
    if args.library == "pydantic_ai":
        from awesome_gradio_agents.doc_qa.pydantic_ai import (
            handle_retry,
            select_data,
            stream_from_agent,
            undo,
        )
    elif args.library == "transformers":
        from awesome_gradio_agents.doc_qa.transformers import (
            handle_retry,
            select_data,
            stream_from_agent,
            undo,
        )
    elif args.library == "langchain":
        from awesome_gradio_agents.doc_qa.langchain import (
            handle_retry,
            select_data,
            stream_from_agent,
            undo,
        )
    else:
        raise ValueError(f"Unknown library {args.library}")

    app = make_app(stream_from_agent, undo, handle_retry, select_data, args.library)
    app.launch()
