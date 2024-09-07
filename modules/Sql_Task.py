import pandas as pd
import numpy as np
import pickle
import os
import re
import logging

from zhipuai import ZhipuAI
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from modelscope import AutoTokenizer, snapshot_download
from utils.sqlite import DB
from utils.config import Config
from utils.prompt import Prompt
from utils.llm import LLM

os.chdir(Config.base_path)

llm = LLM()

class Sql_Task:
    def __init__(self):
        # 加载tokenizer
        self.tokenizer = self.init_tokenizer()
    
        # 加载RAG数据库
        self.frequency_matrix, \
        self.token_to_index, \
        self.questions, \
        self.sqls, \
        self.query_results, \
        self.final_respons = self.init_rag()

        # RAG召回样例个数
        self.top_n = Config.top_n

        # 数据库
        self.db = DB(Config.db_sqlite_url)

    def run(self, new_question):
        # 找出最相似的2-4个问题及其对应的SQL
        exps = self.find_most_similar_questions(new_question)
        q_exps, sql_exps, query_result_exps, final_response_exps = zip(*exps)
        logging.info(q_exps)
        logging.info(sql_exps)

        # 生成初始的SQL
        prompt = Prompt.create_sql_icl_prompt(new_question, zip(q_exps, sql_exps))
        llm_first_respon = llm.get_llm_responses(prompt)

        # 提取初始生成的SQL
        sql_query = self.extract_sql_code(llm_first_respon)

        # 尝试执行SQL查询
        for attempt in range(3):  # 最多尝试3次
            try:
                logging.info(f"尝试 {attempt + 1}: 执行SQL查询: {sql_query}")
                query_result = self.db.select(sql_query)
                logging.info(f"SQL查询成功: {query_result}")

                # 返回提示词
                return Prompt.create_final_icl_prompt(new_question, query_result, zip(q_exps, query_result_exps, final_response_exps)), sql_query

            except Exception as e:
                # 记录错误信息
                error_message = f"{str(e)}"
                logging.error(f"尝试 {attempt + 1} 失败: {error_message}")

                # 生成反思提示并让大模型进行反思
                prompt_head = f"你是一个SQLite数据库专家。\n可供使用的表和字段信息：\n字段：基金代码, 基金全称, 基金简称, 管理人, 托管人, 基金类型, 成立日期, 到期日期, 管理费率, 托管费率, 持仓日期, 股票代码, 股票名称, 数量, 市值, 市值占基金资产净值比, 第N大重仓股, 所在证券市场, 所属国家(地区), 报告类型, 债券类型, 债券名称, 持债数量, 持债市值, 持债市值占基金资产净值比, 对应股票代码, 交易日期, 单位净值, 复权单位净值, 累计单位净值, 资产净值, 昨收盘(元), 今开盘(元), 最高价(元), 最低价(元), 收盘价(元), 成交量(股), 成交金额(元), 行业划分标准, 一级行业名称, 二级行业名称, 公告日期, 截止日期, 报告期期初基金总份额, 报告期基金总申购份额, 报告期基金总赎回份额, 报告期期末基金总份额, 定期报告所属年度, 机构投资者持有的基金份额, 机构投资者持有的基金份额占总份额比例, 个人投资者持有的基金份额, 个人投资者持有的基金份额占总份额比例\n表名：基金基本信息、基金股票持仓明细、基金债券持仓明细、基金可转债持仓明细、基金日行情表、A股票日行情表、港股票日行情表、A股公司行业划分表、基金规模变动表、基金份额持有人结构\n\n请逐步思考并根据下面错误信息改正SQLite语句。"
                prompt = f"{prompt_head}\n错误的sql:{sql_query}\n错误信息:{error_message}\n"
                logging.info(f"生成反思提示: {prompt}")

                # 获取大模型反思后的SQL查询
                llm_rethink_respon = llm.get_llm_responses(prompt)
                sql_query = self.extract_sql_code(llm_rethink_respon)
                logging.info(f"大模型修正后的SQL查询: {sql_query}")
                
        # 如果所有尝试都失败，抛出最后遇到的错误
        final_error_message = "在经过3次反思尝试后仍未能执行SQL。"
        logging.error(final_error_message)
        raise RuntimeError(final_error_message)

    # 初始化模型和tokenizer
    def init_tokenizer(self, model_name = Config.tokenizer_name):
        model_dir = snapshot_download(model_name)
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 加载RAG数据库
    def init_rag(self, path_csv=Config.sql_template_csv_path):
        if os.path.exists(Config.sql_icl_cache_path):
            # 加载持久化数据
            frequency_matrix, token_to_index, questions, sqls, query_results, final_respons = self.load_persisted_data(Config.sql_icl_cache_path)
        else:
            # 加载样本库
            questions, sqls, query_results, final_respons = self.load_csv(path_csv)
            
            # 对样本库中的问题进行tokenizer编码并统计词频
            token_counts = self.tokenize_and_count(questions)
            frequency_matrix, token_to_index = self.create_token_frequency_matrix(token_counts)
            
            # 保存持久化数据
            self.save_persisted_data(frequency_matrix, token_to_index, questions, sqls, query_results, final_respons)

        return frequency_matrix, token_to_index, questions, sqls, query_results, final_respons
    
    # 读取CSV文件
    def load_csv(self, file_path):
        df = pd.read_csv(file_path, encoding='utf-8')
        questions = df['问题'].tolist()
        sqls = df['SQL'].tolist()
        query_results = df['Query_Result'].tolist()
        final_respons = df['FA'].tolist()

        return questions, sqls, query_results, final_respons

    # 持久化sql icl模板
    def save_persisted_data(self, frequency_matrix, token_to_index, questions, sqls, query_results, final_respons):
        with open(Config.sql_icl_cache_path, 'wb') as file:
            pickle.dump((frequency_matrix, token_to_index, questions, sqls, query_results, final_respons), file)

    # 加载持久化sql icl模板
    def load_persisted_data(self, file_path=Config.sql_icl_cache_path):
        with open(file_path, 'rb') as file:
            frequency_matrix, token_to_index, questions, sqls, query_results, final_respons = pickle.load(file)
        return frequency_matrix, token_to_index, questions, sqls, query_results, final_respons
    
    # 统计token词频
    def tokenize_and_count(self, texts):
        token_counts = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_counts.append(Counter(tokens))
        return token_counts

    # 获取token相似矩阵
    def create_token_frequency_matrix(self, token_counts):
        all_tokens = set(token for count in token_counts for token in count.keys())
        token_to_index = {token: i for i, token in enumerate(all_tokens)}
        
        num_texts = len(token_counts)
        num_tokens = len(all_tokens)
        frequency_matrix = np.zeros((num_texts, num_tokens))
        
        for i, counts in enumerate(token_counts):
            for token, count in counts.items():
                if token in token_to_index:
                    frequency_matrix[i, token_to_index[token]] = count
        
        return frequency_matrix, token_to_index
   
    # 对新问题进行tokenizer编码并统计词频
    def find_most_similar_questions(self, new_question):
        new_question_tokens = self.tokenizer.encode(new_question, add_special_tokens=True)
        new_question_count = Counter(new_question_tokens)
        
        # 构建新问题的词频向量
        new_question_vector = np.zeros((1, len(self.token_to_index)))
        
        for token, count in new_question_count.items():
            if token in self.token_to_index:
                new_question_vector[0, self.token_to_index[token]] = count
        
        # 计算新问题与样本库问题之间的余弦相似度
        similarity_scores = cosine_similarity(new_question_vector, self.frequency_matrix)[0]
        
        # 获取相似度最高的top_n个问题及其对应的SQL语句
        most_similar_indices = similarity_scores.argsort()[-self.top_n:][::-1]
        
        most_similar = [(self.questions[i], self.sqls[i], self.query_results[i], self.final_respons[i]) for i in most_similar_indices]

        return most_similar

    # 从包含SQL代码的字符串中提取实际的SQL语句。
    def extract_sql_code(self, code_block: str) -> str:
        # 使用正则表达式匹配所有 SQL 代码块
        matches = re.findall(r'```sql\s*(.*?)\s*```', code_block, re.DOTALL)
        
        if matches:
            # 提取最后一个 SQL 代码块并去除前后空白字符
            last_sql_code = matches[-1].strip()
            return last_sql_code
        else:
            return code_block
