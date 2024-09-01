from utils.config import Config
from utils.prompt import Prompt
from utils.llm import LLM

llm = LLM()

class Generator:
    def __init__(self):
        pass

    def run(self, new_question, query_result, q_exps, query_result_exps, final_response_exps):
        # 拼接回答
        prompt = Prompt.create_final_icl_prompt(new_question, query_result, zip(q_exps, query_result_exps, final_response_exps))
        
        return llm.get_llm_responses(prompt)