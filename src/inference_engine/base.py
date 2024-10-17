import copy
import asyncio
from src.config.constants import DEFAULT_CHAT_PARAMS

class InferenceEngine:
    def __init__(self, model: str = None, **chat_params):
        self.model = model
        self.chat_params = DEFAULT_CHAT_PARAMS
        self.chat_params.update(**chat_params)
        self.setup_client()

    def setup_client(self):
        raise NotImplementedError("setup_client must be implemented in the child class.")

    def update_chat_params(self, chat_params: dict):
        _chat_params = copy.copy(self.chat_params)
        _chat_params.update(chat_params)
        return _chat_params

    def prepare_messages(self, messages: str | list[dict]):
        return messages if isinstance(messages, list) else [{"role": "user", "content": messages}]

    def chat_completions(self, messages: str | list[dict], model: str = None, **chat_params):
        messages = self.prepare_messages(messages)
        chat_params = self.update_chat_params(chat_params)

        response = self.client.chat.completions.create(
            messages=messages,
            model=model or self.model,
            **chat_params,
        )
        return response

class AsyncInferenceEngine(InferenceEngine):
    async def batch_chat_completions(
        self,
        batch_messages:list[list[dict]],
        **kwargs
    ):
        tasks = [self.chat_completions(messages, **kwargs) for messages in batch_messages]
        responses = await asyncio.gather(*tasks)

        return responses
    
