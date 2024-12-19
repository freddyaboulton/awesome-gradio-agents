from enum import Enum


def load_dotenv():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass


class Logo(str, Enum):
    pydantic_ai = "https://ai.pydantic.dev/img/logo-white.svg"
    transformers = "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"
    langchain = (
        "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/svg/1f99c.svg"
    )
