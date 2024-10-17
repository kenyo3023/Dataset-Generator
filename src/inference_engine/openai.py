import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from src.inference_engine.base import InferenceEngine, AsyncInferenceEngine

load_dotenv()

class OpenAIInferenceEngine(InferenceEngine):
    def setup_client(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
class AsyncOpenAIInferenceEngine(AsyncInferenceEngine):
    def setup_client(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))