import os
import asyncio
from dotenv import load_dotenv
from fireworks.client import Fireworks, AsyncFireworks
from src.inference_engine.base import InferenceEngine, AsyncInferenceEngine

load_dotenv()

class FireWorksInferenceEngine(InferenceEngine):
    def __init__(self, model: str, **chat_params):
        prefixed_model = f"accounts/fireworks/models/{model}"
        super().__init__(model=prefixed_model, **chat_params)
       
    def setup_client(self):
        self.client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
    
class AsyncFireWorksInferenceEngine(InferenceEngine):
    def __init__(self, model: str, **chat_params):
        prefixed_model = f"accounts/fireworks/models/{model}"
        super().__init__(model=prefixed_model, **chat_params)

    def setup_client(self):
        self.client = AsyncFireworks(api_key=os.getenv("FIREWORKS_API_KEY"))