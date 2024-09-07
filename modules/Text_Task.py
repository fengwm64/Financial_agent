import os
import glob
import jieba
import pickle
import logging
import numpy as np
import faiss
import tqdm
import warnings

from tqdm import tqdm
from collections import Counter
from rank_bm25 import BM25Okapi
from langchain.text_splitter import MarkdownTextSplitter
from utils.config import Config
from utils.prompt import Prompt
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 关闭特定类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DocumentSearch:
    def __init__(self):
        self.stopwords = self.load_stopwords(Config.stop_word_path)
        self.md_path = Config.pdf_to_md_path
        self.faiss_index_dir = Config.faiss_index_dir
        self.documents = self.load_documents()
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.emb_model_path)
        self.index = self.build_or_load_faiss_index()

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

    # 加载 FAISS 索引
    def build_or_load_faiss_index(self):
        # 检查 FAISS 索引目录是否存在
        if not os.path.exists(self.faiss_index_dir):
            os.makedirs(self.faiss_index_dir)
            print("FAISS index directory created.")
            self.build_faiss_index()
        else:
            # 加载所有文件的 FAISS 索引
            print("Loading FAISS indexes...")
            index = {}
            files_in_dir = os.listdir(self.faiss_index_dir)
            index_files = {file_name for file_name in files_in_dir if file_name.endswith('.index')}
            document_files = {doc['file_name'] for doc in self.documents}
            expected_files = {f"{file_name[:-3]}.index" for file_name in document_files}
            
            # 检查是否缺少文件
            missing_files = expected_files - index_files
            if missing_files:
                print(f"Missing FAISS index files: {missing_files}")
                self.build_faiss_index(missing_files)  # 只构建缺失的索引
                # 重新加载索引
                print("Reloading FAISS indexes...")
                for file_name in tqdm(os.listdir(self.faiss_index_dir), desc="Reloading", unit="file"):
                    if file_name.endswith('.index'):
                        file_path = os.path.join(self.faiss_index_dir, file_name)
                        index[file_name] = faiss.read_index(file_path)
            else:
                # 加载所有现有的 FAISS 索引
                print("All FAISS indexes loaded successfully.")
                for file_name in tqdm(index_files, desc="Loading", unit="file"):
                    file_path = os.path.join(self.faiss_index_dir, file_name)
                    index[file_name] = faiss.read_index(file_path)
        
        return index

    # 建立 FAISS 索引
    def build_faiss_index(self, missing_files=None):
        print("Building FAISS indexes...")
        documents_by_file = {}
        
        # 按文件名分组文档
        for doc in self.documents:
            file_name = doc['file_name']
            if file_name not in documents_by_file:
                documents_by_file[file_name] = []
            documents_by_file[file_name].append(doc['content'])
        
        if missing_files is None:
            # 如果没有指定缺失文件，则构建所有文件的索引
            files_to_process = documents_by_file.keys()
        else:
            # 只构建缺失的索引
            files_to_process = {file_name for file_name in documents_by_file if f"{file_name[:-3]}.index" in missing_files}
        
        # 创建和保存每个文件的 FAISS 索引
        for file_name in tqdm(files_to_process, desc="Building", unit="file"):
            texts = documents_by_file[file_name]
            # 创建 HuggingFaceEmbeddings 对象
            embeddings = self.embeddings

            # 生成文档的嵌入向量
            embedding_vectors = []

            # 使用 tqdm 显示进度条
            for text in tqdm(texts, desc=f"{file_name}", unit="text"):
                vector = embeddings.embed_query(text)
                embedding_vectors.append(vector)

            embedding_vectors = np.array(embedding_vectors, dtype=np.float32)

            # 创建 FAISS 索引
            dimension = embedding_vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embedding_vectors)

            # 保存 FAISS 索引
            file_index_path = os.path.join(self.faiss_index_dir, f"{file_name[:-3]}.index")
            faiss.write_index(index, file_index_path)
            print(f"FAISS index for {file_name} saved to {file_index_path}")

    # 根据查询在文档中进行BM25搜索，并返回相关文档的片段
    def find_most_comm_file(self, query, top_n=5, model_filename=Config.bm25_cache_path, train_if_not_exists=True):
        # 尝试加载现有的 BM25 模型
        try:
            with open(model_filename, 'rb') as file:
                self.bm25 = pickle.load(file)
        except FileNotFoundError:
            if train_if_not_exists:
                # 如果模型文件不存在，则训练并保存模型
                tokenized_docs = [self.tokenize_and_remove_stopwords(doc['content']) for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                with open(model_filename, 'wb') as file:
                    pickle.dump(self.bm25, file)
            else:
                raise FileNotFoundError(f"Model file '{model_filename}' not found and training is disabled.")

        # 确保模型已加载
        if not hasattr(self, 'bm25'):
            raise ValueError("BM25 model is not loaded. Please provide a model filename.")

        # 处理查询
        tokenized_query = self.tokenize_and_remove_stopwords(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            doc = self.documents[i]
            content = doc["content"]
            file_name = doc["file_name"]

            results.append({
                "file_name": file_name,
                "content": content,
                "score": score
            })

        # 排序并返回前 top_n 个结果
        top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

        # 统计文件名出现次数
        file_names = [result['file_name'] for result in top_results]
        most_common_file = Counter(file_names).most_common(1)[0][0]

        return top_results, most_common_file

    def bm25_query_in_file(self, file_name, keyword, top_n=20):
        # 提取指定文件的所有文档片段
        file_docs = [doc for doc in self.documents if doc["file_name"] == file_name+".md"]

        # 文件名用于持久化
        tokenized_docs_file = f"{Config.tokenized_docs_cache_dir}/{file_name}.pkl"

        # 尝试加载持久化的 tokenized_docs
        if os.path.exists(tokenized_docs_file):
            with open(tokenized_docs_file, "rb") as f:
                tokenized_docs = pickle.load(f)
        else:
            # 对文档片段进行分词和去除停用词
            tokenized_docs = [self.tokenize_and_remove_stopwords(doc['content']) for doc in file_docs]
            # 持久化 tokenized_docs
            with open(tokenized_docs_file, "wb") as f:
                pickle.dump(tokenized_docs, f)

        # 对文档片段进行BM25索引
        bm25 = BM25Okapi(tokenized_docs)

        # 对关键词进行BM25搜索
        tokenized_query = self.tokenize_and_remove_stopwords(keyword)
        scores = bm25.get_scores(tokenized_query)

        # 归一化BM25分数到百分制
        if len(scores) > 0:
            max_score = max(scores)
            min_score = min(scores)
            range_score = max_score - min_score

            # 避免除以零的情况
            if range_score == 0:
                normalized_scores = [100] * len(scores)
            else:
                normalized_scores = [100 * (score - min_score) / range_score for score in scores]
        else:
            normalized_scores = []

        # 获取前top_n个匹配的内容
        top_indices = np.argsort(normalized_scores)[-top_n:][::-1]
        top_results = [{"content": file_docs[i]["content"], "score": normalized_scores[i]} for i in top_indices]

        return top_results

    # 使用faiss搜索
    def faiss_query_in_file(self, file_name, keyword, top_n=20):
        # 从索引字典中获取指定文件的索引
        index = self.index.get(file_name)
        if index is None:
            index = self.index.get(file_name + ".index")
            if index is None:
                raise ValueError(f"No FAISS index found for file {file_name}")
        
        # 生成查询文本的嵌入向量
        query_vector = self.embeddings.embed_query(keyword)
        query_vector = np.array([query_vector], dtype=np.float32)

        # 执行查询
        distances, indices = index.search(query_vector, k=top_n)

        # 获取对应的文档内容
        document_sections = [doc['content'] for doc in self.documents if doc['file_name'] == file_name + ".md"]
        print(len(document_sections))
    
        # 创建一个映射来通过索引访问文档内容
        content_map = {i: section for i, section in enumerate(document_sections)}

        # 提取查询结果的片段内容
        results = []
        scores = [1 / (1 + dist) for dist in distances[0]]  # 计算分数

        # 归一化分数到百分制
        if len(scores) > 0:
            max_score = max(scores)
            min_score = min(scores)
            range_score = max_score - min_score

            # 避免除以零的情况
            if range_score == 0:
                normalized_scores = [100] * len(scores)
            else:
                normalized_scores = [100 * (score - min_score) / range_score for score in scores]
        else:
            normalized_scores = []

        # 提取查询结果的片段内容
        for i in range(len(distances[0])):
            idx = indices[0][i]
            # 检查索引是否在 content_map 范围内
            if idx in content_map:
                result = {
                    'content': content_map[idx],
                    'score': float(normalized_scores[i]) if i < len(normalized_scores) else 0.0
                }
                results.append(result)
            else:
                print(f"Index {idx} is out of range for document content.")

        return results

    # 混合搜索
    def hybrid_query_in_file(self, file_name, keyword, top_n=5, bm25_weight=0.5, faiss_weight=0.5):
        # 使用BM25进行查询
        bm25_results = self.bm25_query_in_file(file_name, keyword, top_n)
        
        # 使用FAISS进行查询
        faiss_results = self.faiss_query_in_file(file_name, keyword, top_n)
        
        # 创建结果字典以方便合并
        results_dict = {}

        # 处理BM25结果
        for result in bm25_results:
            content = result["content"]
            score = result["score"] * bm25_weight
            if content in results_dict:
                results_dict[content]["score"] += score
            else:
                results_dict[content] = {"score": score}

        # 处理FAISS结果
        for result in faiss_results:
            content = result["content"]
            score = result["score"] * faiss_weight
            if content in results_dict:
                results_dict[content]["score"] += score
            else:
                results_dict[content] = {"score": score}

        # 将合并的结果转为列表并排序
        merged_results = [{"content": content, "score": result["score"]} for content, result in results_dict.items()]
        top_results = sorted(merged_results, key=lambda x: x["score"], reverse=True)[:top_n]

        return top_results

    # 整合函数
    def run(self, question, company, keyword, top_n=10, bm25_weight=0.5, faiss_weight=0.5):
        # 查找公司所在文件名
        _, most_common_file = self.find_most_comm_file(company, top_n=20)

        logging.info(f"出现次数最多的文件名: {most_common_file}\n")

        # 在文档内查找
        top_results = self.hybrid_query_in_file(most_common_file, keyword, top_n, bm25_weight, faiss_weight)

        if top_results:
            # 提取内容列表
            contents = [result['content'] for result in top_results]
            
            # 拼接提示词
            final_prompt = Prompt.create_final_text_prompt(question, contents)
        else:
            # 抛出异常
            raise ValueError(f"在文件 {most_common_file} 中未找到关键词 '{keyword}'。")
        
        # 返回prompt
        return final_prompt, most_common_file
    
