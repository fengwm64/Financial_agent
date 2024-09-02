import os
import glob
import jieba
import pickle
import logging

from collections import Counter
from rank_bm25 import BM25Okapi
from langchain.text_splitter import MarkdownTextSplitter
from utils.config import Config
from utils.prompt import Prompt

class DocumentSearch:
    def __init__(self, stopword_path, md_path):
        self.stopwords = self.load_stopwords(stopword_path)
        self.md_path = md_path
        self.documents = self.load_documents()
        self.bm25, self.tokenized_docs = self.build_bm25_index()

    def run(self, question, company, keyword):
        # 查找公司所在文件名
        _, most_common_file = self.search_documents(company, top_n=20, context_length=2000)

        logging.info(f"出现次数最多的文件名: {most_common_file}\n")

        # 在文档内查找
        top_results = self.search_in_file(most_common_file, keyword, top_n=10)

        if top_results:
            # 提取内容列表
            contents = [result['content'] for result in top_results]
            
            # 拼接提示词
            final_prompt = Prompt.create_final_text_prompt(question, contents)
            logging.info(f"生成的最终提示词:\n{final_prompt}\n")
        else:
            # 抛出异常
            raise ValueError(f"在文件 {most_common_file} 中未找到关键词 '{keyword}'。")
        
        # 返回prompt
        return final_prompt
    
    # 加载停用词
    def load_stopwords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return set(file.read().splitlines())

    # 加载文档内容
    def load_documents(self):
        if os.path.exists(Config.documents_cache_path):
            with open(Config.documents_cache_path, 'rb') as f:
                return pickle.load(f)

        md_files = glob.glob(os.path.join(self.md_path, "*.md"))
        documents = []
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=200)

        for file_path in md_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                sections = splitter.split_text(content)
                documents.extend([{
                    "file_name": os.path.basename(file_path),
                    "content": section
                } for section in sections])

        with open(Config.documents_cache_path, 'wb') as f:
            pickle.dump(documents, f)

        return documents

    # 分词并去除停用词
    def tokenize_and_remove_stopwords(self, text):
        tokens = jieba.lcut(text)
        return [word for word in tokens if word not in self.stopwords and len(word.strip()) > 1]

    # 建立BM25索引
    def build_bm25_index(self):
        if os.path.exists(Config.bm25_cache_path) and os.path.exists(Config.tokenized_docs_cache_path):
            with open(Config.bm25_cache_path, 'rb') as f:
                bm25 = pickle.load(f)
            with open(Config.tokenized_docs_cache_path, 'rb') as f:
                tokenized_docs = pickle.load(f)
            return bm25, tokenized_docs

        tokenized_docs = [self.tokenize_and_remove_stopwords(doc['content']) for doc in self.documents]
        bm25 = BM25Okapi(tokenized_docs)

        with open(Config.bm25_cache_path, 'wb') as f:
            pickle.dump(bm25, f)
        with open(Config.tokenized_docs_cache_path, 'wb') as f:
            pickle.dump(tokenized_docs, f)

        return bm25, tokenized_docs

    # 根据查询在文档中进行BM25搜索，并返回相关文档的片段
    def search_documents(self, query, top_n=5, context_length=500):
        tokenized_query = self.tokenize_and_remove_stopwords(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            doc = self.documents[i]
            content = doc["content"]
            file_name = doc["file_name"]

            # 生成固定长度的上下文片段
            context_snippet = self.extract_snippets(content, context_length)

            results.append({
                "file_name": file_name,
                "content_snippet": context_snippet,
                "score": score
            })

        # 排序并返回前top_n个结果
        top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

        # 统计文件名出现次数
        file_names = [result['file_name'] for result in top_results]
        most_common_file = Counter(file_names).most_common(1)[0][0]

        return top_results, most_common_file

    # 在特定文件中使用BM25搜索关键词并返回前top_n个结果
    def search_in_file(self, file_name, keyword, top_n=20):
        # 提取指定文件的所有文档片段
        file_docs = [doc for doc in self.documents if doc["file_name"] == file_name]

        # 对文档片段进行BM25索引
        tokenized_docs = [self.tokenize_and_remove_stopwords(doc['content']) for doc in file_docs]
        bm25 = BM25Okapi(tokenized_docs)

        # 对关键词进行BM25搜索
        tokenized_query = self.tokenize_and_remove_stopwords(keyword)
        scores = bm25.get_scores(tokenized_query)

        # 获取前top_n个匹配的内容
        top_indices = scores.argsort()[-top_n:][::-1]
        top_results = [{"content": file_docs[i]["content"], "score": scores[i]} for i in top_indices]

        # 返回前top_n个结果
        return top_results

    # 固定长度的上下文片段
    def extract_snippets(self, content, context_length):
        return content[:context_length]
    