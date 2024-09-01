from zhipuai import ZhipuAI
from utils.config import Config

class LLM:
    def __init__(self):
        self.client = ZhipuAI(api_key=Config.zhipu_api_key)

    # 获取大模型回复
    def get_llm_responses(self, prompt):
            
        response = self.client.chat.completions.create(
            model= Config.base_llm,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content

        return answer