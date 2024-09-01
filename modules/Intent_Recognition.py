from utils.config import Config
from utils.prompt import Prompt
from utils.llm import LLM

llm = LLM()

class Intent_Recognition:
    def __init__(self):
        pass

    def run(self, new_question):
        # 拼接回答
        prompt = Prompt.create_ir_icl_prompt(new_question)
        
        return llm.get_llm_responses(prompt)