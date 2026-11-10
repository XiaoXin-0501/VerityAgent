"""vLLM / llama.cpp Chat adapter（OpenAI 兼容 + continue_final_message）。"""

from .openai_chat import VLLMChatModel, VLLMChatProvider

__all__ = ["VLLMChatModel", "VLLMChatProvider"]
