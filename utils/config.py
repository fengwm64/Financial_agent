import os

# 定义工作目录
WORK_DIR = '/home/fwm/projects/financial_agent/数据处理/SQL/'

# 获取当前文件的上级目录路径
def get_base_file_path():
    return os.path.dirname(os.path.dirname(__file__))

# 配置类，包含各种路径的设置
class Config:
    # 如果 WORK_DIR 没有定义，则使用当前文件的上级目录路径作为基础路径
    if WORK_DIR == '':
        base_path = get_base_file_path()
    else:
        base_path = WORK_DIR

    # 提交的 JSONL 结果文件路径
    res_json_path = os.path.join(base_path, 'out/answer_submit.jsonl')
    
    # 问题的 JSON 文件路径
    question_json_path = os.path.join(base_path, 'data/question.json')
    
    # SQLite 数据库文件路径
    db_sqlite_url = os.path.join(base_path, "data/db/博金杯比赛数据.db").replace("\\", "/")
    
    # SQL 答案模板的 CSV 文件路径
    sql_answer_template_path = os.path.join(base_path, 'data/ICL_SQL.csv')

    # SQL 答案模板嵌入持久化文件路径
    sql_answer_emb_path = os.path.join(base_path, 'cache/ICL_SQL_Data.pkl')

    # md文档划分缓存路径
    doc_cache_path = os.path.join(base_path, 'cache/')

    # API 密钥，用于访问模型或服务（此处为占位符）
    zhipu_api_key = "b2ccc57634d43e3b01b255d5d9b3840d.KNUQg8zN98Id9OwU"

    # 指定使用的分词器名称
    tokenizer_name = "TongyiFinance/Tongyi-Finance-14B-Chat-Int4"

    # md文本链接
    pdf_to_md_path = os.path.join(base_path, "data/pdf_to_md")
    
    # SQL-ICL RAG 模型召回的样例个数
    top_n = 2

    # ICL 使用的大模型名称
    base_llm = "glm-4-0520"

    # 评估日志文件路径
    evaluation_log = os.path.join(base_path, 'log/evaluation_log.log')

    # 停用词文件路径
    stop_word_path = os.path.join(base_path, 'data/stopwords.txt')

    # 词嵌入模型
    emb_model_name = "Alibaba-NLP/gte-multilingual-base"


