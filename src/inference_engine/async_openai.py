import asyncio

import openai

from .openai import OpenAIInferenceEngine


class AsyncOpenAIInferenceEngine(OpenAIInferenceEngine):

    def setup_client(self):
        self.client = openai.AsyncClient()

    async def batch_chat_completions(
        self,
        batch_messages:list[list[dict]],
        **kwargs
    ):
        tasks = [self.chat_completions(messages, **kwargs) for messages in batch_messages]
        responses = await asyncio.gather(*tasks)

        return responses