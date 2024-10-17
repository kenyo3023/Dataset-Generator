import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from src.inference_engine.base import InferenceEngine, AsyncInferenceEngine

load_dotenv()

class OpenRouterInferenceEngine(InferenceEngine):
    def setup_client(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            )
    
class AsyncOpenRouterInferenceEngine(AsyncInferenceEngine):
    def setup_client(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            )