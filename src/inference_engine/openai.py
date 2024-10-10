import os
import copy
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


DEFAULT_CHAT_PARAMS = {
    "max_tokens": 1024,
    "temperature": 0.0,
}

class OpenAIInferenceEngine:

    def __init__(self, model:str=None, **chat_params):
        self.model = model
        self.chat_params = DEFAULT_CHAT_PARAMS
        self.chat_params.update(**chat_params)

    def update_chat_params(self, chat_params:dict):
        _chat_params = chat_params
        chat_params = copy.copy(self.chat_params)
        chat_params.update(_chat_params)
        return chat_params

    def chat_completions(
        self,
        messages:str | list[dict],
        model:str=None,
        **chat_params:None
    ):
        assert model or self.model, "The model was neither specified nor initialized."

        messages = messages if isinstance(messages, list) else [{"role": "user", "content": messages}]
        chat_params = self.update_chat_params(chat_params)

        response = openai.chat.completions.create(
            messages=messages,
            model=model or self.model,
            **chat_params,
        )
        return response