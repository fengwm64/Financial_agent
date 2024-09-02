from utils.config import Config
from utils.prompt import Prompt
from utils.llm import LLM

llm = LLM()

class Generator:
    def __init__(self):
        pass

    def run(self, prompt):
        return llm.get_llm_responses(prompt)