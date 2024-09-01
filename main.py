import jsonlines
import json
import os
import logging

from tqdm import tqdm
from utils.config import Config
from modules.Sql_Task import Sql_Task
from modules.Generator import Generator
from modules.Intent_Recognition import Intent_Recognition

# 配置日志记录
logging.basicConfig(
    filename=Config.evaluation_log,  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 读取jsonl
def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


# 逐个生成问题
def get_question(path_questions=Config.question_json_path):
    querys = read_jsonl(path_questions)
    for item in tqdm(querys):
        query = item["question"]
        yield query

def run_eval(checkpoint_file=Config.res_json_path):
    question_generator = get_question()  # 获取问题生成器
    
    # 如果检查点文件存在，加载之前的结果
    results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            results = [json.loads(line) for line in file.readlines()]

    processed_ids = {result["id"] for result in results}  # 已处理问题的ID集合
    ir = Intent_Recognition()  # 加载意图识别模型
    sql_task = Sql_Task()  # 加载SQL任务执行器
    generator = Generator()  # 加载结果生成器
    sql_error = 0  # 初始化SQL错误计数器

    for i, question in enumerate(question_generator):
        # 跳过已处理的问题
        if i in processed_ids:
            continue
        
        logging.info(f"Processing question {i}: {question}")
        
        query_result = None  # 初始化查询结果变量
        q_exps = query_result_exps = final_response_exps = None  # 初始化解释变量

        # 处理每个问题
        if "数据查询" in ir.run(question):
            logging.info("Detected intent: 数据查询")
            try:
                # 尝试执行SQL任务
                q_exps, query_result, query_result_exps, final_response_exps = sql_task.run(question)
                logging.info(f"SQL Task executed successfully for question {i}")
            except Exception as e:
                # 捕获异常并增加SQL错误计数
                sql_error += 1
                query_result = f"SQL 查询失败: {e}"
                logging.error(f"SQL Task failed for question {i}: {e}")
                continue
        else:
            logging.info("Detected intent: 文本分析")
            # 处理非SQL任务的逻辑（此处为占位符）
            query_result = "文本分析结果"  # 你可以在此处替换为实际的文本分析结果
            continue

        # 生成最终的回答
        answer = generator.run(question, query_result, q_exps, query_result_exps, final_response_exps)
        logging.info(f"Generated answer for question {i}: {answer}")

        # 创建结果字典
        result = {
            "id": i,
            "question": question,
            "answer": answer
        }

        # 逐步将结果写入文件
        with open(checkpoint_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        logging.info(f"Result for question {i} written to checkpoint file.")
        logging.info(f"Total SQL errors so far: {sql_error}")

if __name__ == "__main__":
    run_eval()
