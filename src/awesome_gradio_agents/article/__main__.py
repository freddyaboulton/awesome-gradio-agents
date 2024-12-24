from awesome_gradio_agents.utils import Logo, load_dotenv
import gradio as gr

def make_app(fn, undo_fn, retry_fn, select_fn, library: str):
    with gr.Blocks() as demo:
        gr.Markdown("# 📝 AI Article Writing Crew")
        
        openai_api_key = gr.Textbox(
            label='OpenAI API Key',
            type='password',
            placeholder='Enter your OpenAI API key...',
            interactive=True
        )

        chatbot = gr.Chatbot(
            label="Writing Process",
            height=700,
            show_label=True,
            bubble_full_width=False,
            type="messages",
            render_markdown=True
        )

        topic = gr.Textbox(
            label="Article Topic",
            placeholder="Enter topic...",
            interactive=True
        )

        async def process_input(topic_text, history, api_key):
            if not api_key:
                history = []
                history.append({
                    "role": "assistant",
                    "content": "Please provide an OpenAI API key",
                    "metadata": {"title": "❌ Error"}
                })
                yield topic_text, history, None
                return

            history = history or []
            async for update in fn(topic_text, history, api_key):
                yield update

        topic.submit(process_input, 
                    inputs=[topic, chatbot, openai_api_key],
                    outputs=[topic, chatbot, gr.State()])

        chatbot.retry(retry_fn, [chatbot, gr.State()], [topic, chatbot, gr.State()])
        chatbot.undo(undo_fn, [chatbot, gr.State()], [topic, chatbot, gr.State()])

    return demo

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, default="crewai")
    args = parser.parse_args()
    load_dotenv()

    if args.library == "crewai":
        from awesome_gradio_agents.article.crewai import (
            handle_retry,
            select_data,
            stream_from_agent,
            undo,
        )
    else:
        raise ValueError(f"Unknown library {args.library}")

    app = make_app(stream_from_agent, undo, handle_retry, select_data, args.library)
    app.queue()
    app.launch()