import gradio as gr
import jsonlines
import json
import os
import logging
import re

from modules.Sql_Task import Sql_Task
from modules.Text_Task import DocumentSearch
from modules.Generator import Generator
from modules.Intent_Recognition import Intent_Recognition
from utils.config import Config

os.chdir(Config.base_path)

# 配置日志记录
logging.basicConfig(
    filename='log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content

def process_questions(file):
    checkpoint_file = Config.res_json_path
    results = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f.readlines()]

    processed_ids = {result["id"] for result in results}
    ir = Intent_Recognition()
    sql_task = Sql_Task()
    generator = Generator()
    ds = DocumentSearch(
        stopword_path=Config.stop_word_path,
        md_path=Config.pdf_to_md_path
    )
    sql_error = 0
    output = []

    querys = read_jsonl(file.name)
    for i, item in enumerate(querys):
        if i in processed_ids:
            continue
        
        question = item["question"]
        logging.info(f"Processing question {i}: {question}")

        classfly = ir.run(question)
        logging.info(classfly)

        prompt = None
        if "数据查询" in classfly and "文本理解" not in classfly:
            logging.info("Detected intent: 数据查询")
            try:
                prompt, _ = sql_task.run(question)
                logging.info(f"SQL Task executed successfully for question {i}")
                logging.info(f"prompt {prompt}")
            
            except Exception as e:
                sql_error += 1
                query_result = f"SQL 查询失败: {e}"
                logging.error(f"SQL Task failed for question {i}: {e}")
                continue
        elif "文本理解" in classfly and "数据查询" not in classfly:
            logging.info("Detected intent: 文本理解")
            company_name = re.search(r'公司名称：\s*[“"\'\(《【{]*([^”"\'\)》】}]+)[”"\'\)》】}]*', classfly).group(1)
            keyword = re.search(r'关键词：\s*[“"\'\(《【{]*([^”"\'\)》】}]+)[”"\'\)》】}]*', classfly).group(1)

            logging.info("公司名称:"+company_name)
            logging.info("关键词:"+keyword)
            
            if company_name is not None and keyword is not None:
                prompt, md_file = ds.run(question, company_name, keyword)

            logging.info(f"prompt {prompt}")
            logging.info(f"md file: {md_file}")
        
        else:
            continue

        answer = generator.run(prompt)
        logging.info(f"Generated answer for question {i}: {answer}")

        result = {
            "id": i,
            "question": question,
            "classification": classfly,
            "prompt": prompt,
            "md_file": md_file if "文本理解" in classfly else None,
            "answer": answer
        }

        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        output.append(result)
        logging.info(f"Result for question {i} written to checkpoint file.")
        logging.info(f"Total SQL errors so far: {sql_error}")

    return output

def gradio_interface(file):
    results = process_questions(file)
    return results

def chat_interface(input_text):
    ir = Intent_Recognition()
    sql_task = Sql_Task()
    generator = Generator()
    ds = DocumentSearch(
        stopword_path=Config.stop_word_path,
        md_path=Config.pdf_to_md_path
    )

    logging.info(f"Processing question: {input_text}")

    classfly = ir.run(input_text)
    logging.info(classfly)

    prompt = None
    sql = None
    md_file = None
    company_name = None
    keyword = None

    if "数据查询" in classfly and "文本理解" not in classfly:
        logging.info("Detected intent: 数据查询")
        try:
            prompt, sql = sql_task.run(input_text)
            logging.info("SQL Task executed successfully.")
            logging.info(f"SQL Query: {sql}")
        except Exception as e:
            logging.error(f"SQL Task failed: {e}")
            return [[input_text, f"SQL 查询失败: {e}"]], f"任务类型: 数据查询\nSQL 查询失败: {e}"

    elif "文本理解" in classfly and "数据查询" not in classfly:
        logging.info("Detected intent: 文本理解")
        company_name = re.search(r'公司名称：\s*[“"\'\(《【{]*([^”"\'\)》】}]+)[”"\'\)》】}]*', classfly).group(1)
        keyword = re.search(r'关键词：\s*[“"\'\(《【{]*([^”"\'\)》】}]+)[”"\'\)》】}]*', classfly).group(1)

        logging.info("公司名称:" + company_name)
        logging.info("关键词:" + keyword)
        
        if company_name and keyword:
            prompt, md_file = ds.run(input_text, company_name, keyword)
            logging.info(f"Markdown file: {md_file}")

    else:
        return [[input_text, "未识别的意图"]], "任务类型: 未识别的意图"

    answer = generator.run(prompt) if prompt else "无法生成回答"
    logging.info(f"Generated answer: {answer}")

    if sql:
        task_info = f"**任务类型:** 数据查询\n\n**SQL 语句:**\n```\n{sql}\n```"
    elif md_file:
        task_info = (
            f"**任务类型:** 文本理解\n\n"
            f"**公司名称:** {company_name}\n\n"
            f"**关键词:** {keyword}\n\n"
            f"**Markdown 文件:** {md_file}"
        )
    else:
        task_info = "**任务类型:** 无法生成回答"

    return [[input_text, answer]], task_info


with gr.Blocks() as demo:
    gr.Markdown("## 任务评测系统")

    with gr.Column():
        chatbot = classification_label = None
        with gr.Row():
            chatbot = gr.Chatbot()
            classification_label = gr.Markdown(label="任务分类及信息")

        msg = gr.Textbox(label="输入你的问题")
        send_btn = gr.Button("发送")
                
        # Adjust the `send_btn.click` and `msg.submit` to update both the chatbot and the classification label
        send_btn.click(chat_interface, inputs=msg, outputs=[chatbot, classification_label])
        msg.submit(chat_interface, inputs=msg, outputs=[chatbot, classification_label])

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传 JSONL 文件")
            evaluate_button = gr.Button("开始评测")
            result_output = gr.Dataframe(headers=["ID", "Question", "Classification", "Prompt/MD File", "Answer"], type="pandas")

            evaluate_button.click(
                gradio_interface, 
                inputs=file_input, 
                outputs=result_output
            )

if __name__ == "__main__":
    demo.launch()
