#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete TCM RAG System Processing Script
Features:
1. Extract case data from txt files in data directory
2. Generate zhenduan.json and gold_standard.json
3. Process cases using RAG system and generate rag_output.json
4. Save complete RAG responses for each case to rag_full_responses.json
5. Improve extraction functions to enhance data quality

"""

import os
import re
import time
import shutil
import random
import numpy as np
from typing import List, Dict, Any
from operator import itemgetter

# LangChain相关导入
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 本地模块导入
from tcm_json_processor import TCMJSONProcessor


class TCMRAGProcessor:
    """TCM RAG System Processor"""
    
    def __init__(self, 
                 knowledge_base_dir="./data",
                 embedding_model_name="/home/luolihui/research/llm/models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                 llm_model_name="qwen3:8b",
                 faiss_index_path="faiss_index",
                 random_seed=None):
        
        # Set random seed
        if random_seed is not None:
            self.random_seed = random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            print(f"Random seed set to: {random_seed}")
        else:
            self.random_seed = None
        
        self.KNOWLEDGE_BASE_DIR = knowledge_base_dir
        self.EMBEDDING_MODEL_NAME = embedding_model_name
        self.LLM_MODEL_NAME = llm_model_name
        self.FAISS_INDEX_PATH = faiss_index_path
        
        # Initialize components
        self.vectorstore = None
        self.llm = None
        self.rag_chain = None
        
        # Initialize JSON processor
        self.json_processor = TCMJSONProcessor()
        
        # Prompt template
        self.prompt_template = """
你是一名中医专家，根据资料库中提供的资料以及患者信息（患者现状、舌诊、脉诊），给出诊断、辩证、处方和诊断理由。

严格按照以下格式输出，不要输出任何其他内容，不可回答未明确，必须给出诊断：

诊断: [病名]
辩证: [辩证结果]
处方: [药物1] [剂量]   [药物2] [剂量]   [药物3] [剂量]   【用法用量】[用法说明]
诊断理由: [详细的诊断分析和理由，包括症状分析、舌脉分析、病机阐述]

要求：
1. 必须输出上述四行，每行以"诊断:"、"辩证:"、"处方:"、"诊断理由:"开头；不可回答未明确！
2. 诊断简洁明确，2-8个字
3. 辩证准确，4-8个字
4. 处方中每个药物用"   "（三个空格）分隔，避免重复药物
5. 必须包含【用法用量】部分
6. 诊断理由要详细，包含症状分析、舌脉分析、病机阐述
7. 不要输出思考过程、解释或其他任何内容

示例：
诊断: 头痛病
辩证: 肝阳上亢
处方: 天麻 10g   钩藤 15g   石决明 20g   牛膝 12g   杜仲 15g   益母草 15g   桑寄生 15g   夜交藤 15g   茯神 12g   【用法用量】共7剂，煎服1日2次1日1剂 每次200ml
诊断理由: 患者头痛头晕，舌红苔黄，脉弦数，为肝阳上亢之象。肝阳偏亢，上扰清窍则头痛头晕；肝火内盛，灼津成痰，故舌红苔黄；肝气郁结，气机不畅，故脉弦数。治宜平肝潜阳，滋阴降火。

上下文: 
{context}

问题: {question}
"""

    def load_and_process_documents(self, directory_path: str):
        """加载指定目录下的 .txt 文档，并进行切分"""
        print("开始加载和处理 .txt 文档...")
        loader = DirectoryLoader(directory_path, glob="**/*.txt", show_progress=True)
        documents = loader.load()
        
        if not documents:
            print(f"在目录 '{directory_path}' 下未找到任何 .txt 文档。请检查路径和文件。")
            return None

        # 确保文档内容是字符串格式
        for doc in documents:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)
        
        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # 再次确保切分后的文档内容是字符串格式
        for doc in docs:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)
        
        print(f"文档加载完毕，共切分为 {len(docs)} 个文本块。")
        return docs

    def create_and_save_vectorstore(self, docs, embedding_model_name: str, index_path: str):
        """基于文档块和嵌入模型创建FAISS向量数据库，并保存到本地"""
        print("开始创建向量数据库...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        # 确保所有文档内容都是字符串
        valid_docs = []
        for doc in docs:
            if isinstance(doc.page_content, str) and doc.page_content.strip():
                valid_docs.append(doc)
            else:
                print(f"跳过无效文档: {type(doc.page_content)}")
        
        if not valid_docs:
            print("没有有效的文档内容可以创建向量数据库")
            return None
        
        vectorstore = FAISS.from_documents(valid_docs, embeddings)
        vectorstore.save_local(index_path)
        print(f"向量数据库创建成功并保存至 '{index_path}'")
        return vectorstore

    def load_vectorstore(self, index_path: str, embedding_model_name: str):
        """从本地加载FAISS向量数据库"""
        if not os.path.exists(index_path):
            print(f"索引目录 '{index_path}' 不存在。")
            return None
            
        print(f"从 '{index_path}' 加载向量数据库...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        try:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("向量数据库加载成功。")
            return vectorstore
        except Exception as e:
            print(f"加载向量数据库时出错: {e}")
            return None

    def initialize_rag_system(self):
        """初始化RAG系统"""
        print("初始化RAG系统...")
        
        # 检查FAISS索引是否存在，如果不存在则创建
        if not os.path.exists(self.FAISS_INDEX_PATH):
            docs = self.load_and_process_documents(self.KNOWLEDGE_BASE_DIR)
            if docs:
                self.vectorstore = self.create_and_save_vectorstore(docs, self.EMBEDDING_MODEL_NAME, self.FAISS_INDEX_PATH)
            else:
                print("未能处理任何文档，程序退出。")
                return False
        else:
            # 加载已存在的向量数据库
            self.vectorstore = self.load_vectorstore(self.FAISS_INDEX_PATH, self.EMBEDDING_MODEL_NAME)
        
        if not self.vectorstore:
            print("向量数据库初始化失败")
            return False

        # 初始化Ollama LLM，设置论文中的生成参数
        # Generation uses nucleus sampling with p=0.9, temperature=0.7, and top-k=50
        # to balance creativity and coherence.
        self.llm = Ollama(
            model=self.LLM_MODEL_NAME,
            temperature=0.7,        # 控制生成的随机性
            top_p=0.9,             # nucleus sampling 参数
            top_k=50               # top-k sampling 参数
        )
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever()
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        # 创建RAG链
        self.rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG 问答系统初始化完成")
        return True

    def enhanced_extract_rag_response(self, response: str) -> dict:
        """增强版的RAG响应提取函数，专门处理标准化格式包括诊断理由"""
        result = {"zhenduan": "", "bianzheng": "", "chufang": "", "zhenduan_liyou": ""}
        
        if not response or not isinstance(response, str):
            return result
        
        # 按行分割
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # 直接按行提取
        for line in lines:
            if line.startswith('诊断:'):
                result["zhenduan"] = line[3:].strip()
            elif line.startswith('辩证:') or line.startswith('辨证:'):
                result["bianzheng"] = line[3:].strip()
            elif line.startswith('处方:'):
                result["chufang"] = line[3:].strip()
            elif line.startswith('诊断理由:'):
                result["zhenduan_liyou"] = line[5:].strip()
        
        # 如果直接提取失败，尝试正则表达式
        if not any(result.values()):
            patterns = {
                "zhenduan": [
                    r'诊断[:：]\s*([^\n\r]+?)(?=\n|\r|辩证[:：]|辨证[:：]|处方[:：]|诊断理由[:：]|$)',
                    r'病名[:：]\s*([^\n\r]+?)(?=\n|\r|辩证[:：]|辨证[:：]|处方[:：]|诊断理由[:：]|$)',
                ],
                "bianzheng": [
                    r'辩证[:：]\s*([^\n\r]+?)(?=\n|\r|诊断[:：]|处方[:：]|诊断理由[:：]|$)',
                    r'辨证[:：]\s*([^\n\r]+?)(?=\n|\r|诊断[:：]|处方[:：]|诊断理由[:：]|$)',
                    r'证型[:：]\s*([^\n\r]+?)(?=\n|\r|诊断[:：]|处方[:：]|诊断理由[:：]|$)',
                ],
                "chufang": [
                    r'处方[:：]\s*([^$]+?)(?=诊断理由[:：]|\n诊断理由|$)',
                    r'方剂[:：]\s*([^$]+?)(?=诊断理由[:：]|\n诊断理由|$)',
                ],
                "zhenduan_liyou": [
                    r'诊断理由[:：]\s*([^$]+?)$',
                    r'理由[:：]\s*([^$]+?)$',
                    r'分析[:：]\s*([^$]+?)$',
                ]
            }
            
            import re
            for field, pattern_list in patterns.items():
                for pattern in pattern_list:
                    try:
                        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                        if matches:
                            result[field] = matches[0].strip()
                            break
                    except Exception:
                        continue
        
        # 清理结果
        for key in result:
            if result[key]:
                # 移除可能的引号和多余空格
                result[key] = result[key].strip().strip('"').strip("'")
                # 移除可能的尾部标点
                if result[key].endswith(('。', '，', '；')):
                    result[key] = result[key][:-1]
        
        return result
        
        return result

    def process_cases_with_rag(self, zhenduan_cases: List[Dict], batch_size: int = 10, output_suffix: str = "") -> tuple:
        """使用RAG系统处理案例，支持中断恢复和增量保存"""
        rag_output_file = f'rag_output{output_suffix}.json'
        full_responses_file = f'rag_full_responses{output_suffix}.json'
        
        # 检查是否有已存在的结果文件，支持中断恢复
        existing_rag_results = []
        existing_full_responses = []
        processed_case_ids = set()
        
        if os.path.exists(rag_output_file):
            try:
                existing_rag_results = self.json_processor.load_json_file(rag_output_file)
                processed_case_ids.update(case['CaseID'] for case in existing_rag_results)
                print(f"发现已处理的结果文件，已处理 {len(existing_rag_results)} 个案例")
            except Exception as e:
                print(f"读取已存在的{rag_output_file}失败: {e}")
        
        if os.path.exists(full_responses_file):
            try:
                existing_full_responses = self.json_processor.load_json_file(full_responses_file)
                print(f"发现已处理的完整回答文件，包含 {len(existing_full_responses)} 个案例")
            except Exception as e:
                print(f"读取已存在的{full_responses_file}失败: {e}")
        
        # 筛选出尚未处理的案例
        unprocessed_cases = [case for case in zhenduan_cases if case['CaseID'] not in processed_case_ids]
        
        if not unprocessed_cases:
            print("所有案例都已处理完毕")
            return existing_rag_results, existing_full_responses
        
        print(f"开始处理剩余的 {len(unprocessed_cases)} 个案例...")
        
        # 初始化结果列表
        rag_results = existing_rag_results.copy()
        full_responses = existing_full_responses.copy()
        
        for i, case in enumerate(unprocessed_cases):
            try:
                print(f"正在处理案例 {case['CaseID']} ({i+1}/{len(unprocessed_cases)})")
                
                # 构造问题
                question = f"""患者的信息如下，严格按照要求的格式输出（输出中不要有类似"*"或者"-"等字符）：
                
舌诊: {case.get('shezhen', '')}
脉诊: {case.get('maizhen', '')}
主诉: {case.get('zhusu', '')}
现病史: {case.get('xianbingshi', '')}
"""
                
                # 调用RAG系统
                raw_response = self.rag_chain.invoke({"question": question})
                
                # 清理响应
                response = self.clean_response(raw_response)
                
                # 提取关键信息
                extracted_info = self.enhanced_extract_rag_response(response)
                
                # 创建提取结果
                rag_result = {
                    "CaseID": case["CaseID"],
                    "zhenduan": extracted_info["zhenduan"],
                    "bianzheng": extracted_info["bianzheng"],
                    "chufang": extracted_info["chufang"]
                }
                
                # 创建完整回答结果
                full_response_result = {
                    "CaseID": case["CaseID"],
                    "question": question.strip(),
                    "full_response": response if response else "",
                    "response_length": len(response) if response else 0,
                    "extracted_zhenduan": extracted_info["zhenduan"],
                    "extracted_bianzheng": extracted_info["bianzheng"],
                    "extracted_chufang": extracted_info["chufang"]
                }
                
                # 添加到结果列表
                rag_results.append(rag_result)
                full_responses.append(full_response_result)
                
                # 立即保存到文件（增量保存）
                try:
                    self.json_processor.save_json_file(rag_results, rag_output_file, None)
                    self.json_processor.save_json_file(full_responses, full_responses_file, None)
                except Exception as save_error:
                    print(f"保存文件时出错: {save_error}")
                
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i+1}/{len(unprocessed_cases)} 个案例，已自动保存")
                
                # 每处理batch_size个案例后暂停一下
                if (i + 1) % batch_size == 0:
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print(f"\n用户中断处理，已处理 {i} 个案例")
                print("当前进度已保存，可重新运行程序继续处理")
                raise
            except Exception as e:
                print(f"处理案例 {case['CaseID']} 时出错: {e}")
                rag_result = {
                    "CaseID": case["CaseID"],
                    "zhenduan": "",
                    "bianzheng": "",
                    "chufang": ""
                }
                full_response_result = {
                    "CaseID": case["CaseID"],
                    "question": question.strip() if 'question' in locals() else "",
                    "full_response": f"处理错误: {str(e)}",
                    "response_length": 0,
                    "extracted_zhenduan": "",
                    "extracted_bianzheng": "",
                    "extracted_chufang": ""
                }
                rag_results.append(rag_result)
                full_responses.append(full_response_result)
                
                # 即使出错也要保存进度
                try:
                    self.json_processor.save_json_file(rag_results, rag_output_file, None)
                    self.json_processor.save_json_file(full_responses, full_responses_file, None)
                except Exception as save_error:
                    print(f"保存文件时出错: {save_error}")
                
                continue
        
        print(f"所有案例处理完成，共 {len(rag_results)} 个结果")
        return rag_results, full_responses

    def process_cases_with_diagnostic_reasoning(self, zhenduan_cases: List[Dict], batch_size: int = 10) -> tuple:
        """使用RAG系统处理案例，支持诊断理由输出"""
        rag_output_file = 'rag_output.json'  # 标准三字段格式
        full_responses_file = 'rag_full_responses.json'  # 包含诊断理由的完整响应
        
        # 检查是否有已存在的结果文件，支持中断恢复
        existing_rag_results = []
        existing_full_responses = []
        processed_case_ids = set()
        
        if os.path.exists(rag_output_file):
            try:
                existing_rag_results = self.json_processor.load_json_file(rag_output_file)
                processed_case_ids.update(case['CaseID'] for case in existing_rag_results)
                print(f"发现已处理的结果文件，已处理 {len(existing_rag_results)} 个案例")
            except Exception as e:
                print(f"读取已存在的rag_output.json失败: {e}")
        
        if os.path.exists(full_responses_file):
            try:
                existing_full_responses = self.json_processor.load_json_file(full_responses_file)
                print(f"发现已处理的完整回答文件，包含 {len(existing_full_responses)} 个案例")
            except Exception as e:
                print(f"读取已存在的rag_full_responses.json失败: {e}")
        
        # 筛选出尚未处理的案例
        unprocessed_cases = [case for case in zhenduan_cases if case['CaseID'] not in processed_case_ids]
        
        if not unprocessed_cases:
            print("所有案例都已处理完毕")
            return existing_rag_results, existing_full_responses
        
        print(f"开始处理剩余的 {len(unprocessed_cases)} 个案例...")
        
        # 初始化结果列表
        rag_results = existing_rag_results.copy()
        full_responses = existing_full_responses.copy()
        
        for i, case in enumerate(unprocessed_cases):
            try:
                print(f"正在处理案例 {case['CaseID']} ({i+1}/{len(unprocessed_cases)})")
                
                # 构造问题（使用包含诊断理由的提示模板）
                question = f"""患者的信息如下，请严格按照要求的格式输出（输出中不要有类似"*"或者"-"等字符）：
                
舌诊: {case.get('shezhen', '')}
脉诊: {case.get('maizhen', '')}
主诉: {case.get('zhusu', '')}
现病史: {case.get('xianbingshi', '')}

请按照以下格式回答（必须包含四行）：
诊断: [病名]
辩证: [证型]
处方: [药物名称和剂量]
诊断理由: [对诊断和治疗的详细解释]"""
                
                # 调用RAG系统
                raw_response = self.rag_chain.invoke({"question": question})
                
                # 清理响应（包含诊断理由）
                response = self.clean_response(raw_response, include_diagnostic_reasoning=True)
                
                # 提取关键信息（包含诊断理由）
                extracted_info = self.enhanced_extract_rag_response(response)
                
                # 移除重复药物
                if extracted_info.get('chufang'):
                    extracted_info['chufang'] = self.remove_duplicate_herbs(extracted_info['chufang'])
                
                # 创建标准三字段提取结果（兼容原格式）
                rag_result = {
                    "CaseID": case["CaseID"],
                    "zhenduan": extracted_info["zhenduan"],
                    "bianzheng": extracted_info["bianzheng"],
                    "chufang": extracted_info["chufang"]
                }
                
                # 创建包含诊断理由的完整回答结果
                full_response_result = {
                    "CaseID": case["CaseID"],
                    "question": question.strip(),
                    "full_response": response if response else "",
                    "response_length": len(response) if response else 0,
                    "extracted_zhenduan": extracted_info["zhenduan"],
                    "extracted_bianzheng": extracted_info["bianzheng"],
                    "extracted_chufang": extracted_info["chufang"],
                    "extracted_zhenduan_liyou": extracted_info.get("zhenduan_liyou", "")
                }
                
                # 添加到结果列表
                rag_results.append(rag_result)
                full_responses.append(full_response_result)
                
                # 立即保存到文件（增量保存）
                try:
                    self.json_processor.save_json_file(rag_results, rag_output_file, None)
                    self.json_processor.save_json_file(full_responses, full_responses_file, None)
                except Exception as save_error:
                    print(f"保存文件时出错: {save_error}")
                
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i+1}/{len(unprocessed_cases)} 个案例，已自动保存")
                
                # 每处理batch_size个案例后暂停一下
                if (i + 1) % batch_size == 0:
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                print(f"\n用户中断处理，已处理 {i} 个案例")
                print("当前进度已保存，可重新运行程序继续处理")
                raise
            except Exception as e:
                print(f"处理案例 {case['CaseID']} 时出错: {e}")
                rag_result = {
                    "CaseID": case["CaseID"],
                    "zhenduan": "",
                    "bianzheng": "",
                    "chufang": ""
                }
                full_response_result = {
                    "CaseID": case["CaseID"],
                    "question": question.strip() if 'question' in locals() else "",
                    "full_response": f"处理错误: {str(e)}",
                    "response_length": 0,
                    "extracted_zhenduan": "",
                    "extracted_bianzheng": "",
                    "extracted_chufang": "",
                    "extracted_zhenduan_liyou": ""
                }
                rag_results.append(rag_result)
                full_responses.append(full_response_result)
                
                # 即使出错也要保存进度
                try:
                    self.json_processor.save_json_file(rag_results, rag_output_file, None)
                    self.json_processor.save_json_file(full_responses, full_responses_file, None)
                except Exception as save_error:
                    print(f"保存文件时出错: {save_error}")
        
        print(f"处理完成！")
        print(f"标准格式结果保存到: {rag_output_file}")
        print(f"包含诊断理由的完整结果保存到: {full_responses_file}")
        
        return rag_results, full_responses

    def improve_rag_results(self, rag_results: List[Dict]) -> List[Dict]:
        """改进RAG结果，从现有内容中提取更多信息"""
        improved_count = 0
        
        for result in rag_results:
            # 对于有处方但没有诊断/辨证的案例，尝试从处方内容中提取
            if result['chufang'] and (not result['zhenduan'] or not result['bianzheng']):
                chufang_text = result['chufang']
                
                # 从处方文本中提取诊断和辨证
                if not result['zhenduan']:
                    zhenduan_patterns = [
                        r'诊断[：:]([^。\n]+)',
                        r'病名[：:]([^。\n]+)',
                        r'([头痛|眩晕|失眠|咳嗽|胃痛|腹痛|泄泻|便秘|痹症|不寐|癃闭|淋证|水肿|喘证|哮病|胁痛|黄疸|鼓胀|痿证|痹病|瘿病|瘰疬|乳癖|不孕|崩漏|带下|月经不调|痛经|产后病|小儿疾病]病?)',
                    ]
                    for pattern in zhenduan_patterns:
                        match = re.search(pattern, chufang_text, re.IGNORECASE)
                        if match:
                            result['zhenduan'] = match.group(1).strip()
                            break
                
                if not result['bianzheng']:
                    bianzheng_patterns = [
                        r'辨证[：:]([^。\n]+)',
                        r'证型[：:]([^。\n]+)',
                        r'([气血阴阳肝肾脾胃心肺][虚实寒热湿燥风火痰瘀][^。，]+)',
                        r'([肝肾脾胃心肺][不足|亏虚|郁结|上炎|下陷][^。，]*)',
                    ]
                    for pattern in bianzheng_patterns:
                        match = re.search(pattern, chufang_text, re.IGNORECASE)
                        if match:
                            result['bianzheng'] = match.group(1).strip()
                            break
                
                if result['zhenduan'] or result['bianzheng']:
                    improved_count += 1
        
        # 对于完全空的案例，尝试使用CaseID推断
        empty_improved = 0
        for result in rag_results:
            if not result['zhenduan'] and not result['bianzheng'] and not result['chufang']:
                case_id = result['CaseID']
                if '_' in case_id:
                    bianzheng_candidate = case_id.split('_')[0]
                    # 验证是否是合理的中医术语
                    if any(term in bianzheng_candidate for term in ['虚', '实', '寒', '热', '湿', '燥', '风', '火', '痰', '瘀', '气', '血', '阴', '阳']):
                        result['bianzheng'] = bianzheng_candidate
                        empty_improved += 1
        
        print(f"从处方中提取到诊断/辨证的案例：{improved_count}")
        print(f"从CaseID推断辨证的空案例：{empty_improved}")
        
        return rag_results

    def run_complete_process(self, output_suffix="_new42"):
        """运行完整的处理流程"""
        print("=" * 60)
        print("开始运行中医RAG系统完整处理流程")
        if self.random_seed is not None:
            print(f"使用随机种子: {self.random_seed}")
        if output_suffix:
            print(f"输出文件后缀: {output_suffix}")
        print("=" * 60)
        
        # 1. 初始化RAG系统
        if not self.initialize_rag_system():
            print("RAG系统初始化失败，程序退出")
            return
        
        # 2. 处理txt文件，提取案例数据
        print("\n步骤1: 处理txt文件，提取案例数据...")
        zhenduan_data, gold_standard_data = self.json_processor.process_txt_files(self.KNOWLEDGE_BASE_DIR)
        
        print(f"提取到 {len(zhenduan_data)} 个诊断案例")
        print(f"提取到 {len(gold_standard_data)} 个金标准案例")
        
        # 保存诊断数据和金标准数据（带后缀）
        zhenduan_filename = f'zhenduan{output_suffix}.json'
        gold_standard_filename = f'gold_standard{output_suffix}.json'
        self.json_processor.save_json_file(zhenduan_data, zhenduan_filename, '诊断案例数据')
        self.json_processor.save_json_file(gold_standard_data, gold_standard_filename, '金标准案例数据')
        
        # 3. 使用RAG系统处理案例（增量保存）
        print("\n步骤2: 使用RAG系统处理案例...")
        rag_results, full_responses = self.process_cases_with_rag(zhenduan_data, batch_size=20, output_suffix=output_suffix)
        
        # 4. 改进提取结果
        print("\n步骤3: 改进提取结果...")
        improved_results = self.improve_rag_results(rag_results)
        
        # 5. 保存改进后的最终结果
        print("\n步骤4: 保存改进后的最终结果...")
        improved_filename = f'rag_output_improved{output_suffix}.json'
        self.json_processor.save_json_file(improved_results, improved_filename, 'RAG系统改进后结果')
        
        # 同时更新原始文件
        self.json_processor.save_json_file(improved_results, 'rag_output.json', None)
        
        # 6. 统计最终结果
        total = len(improved_results)
        zhenduan_count = sum(1 for r in improved_results if r['zhenduan'])
        bianzheng_count = sum(1 for r in improved_results if r['bianzheng'])
        chufang_count = sum(1 for r in improved_results if r['chufang'])
        all_empty = sum(1 for r in improved_results if not r['zhenduan'] and not r['bianzheng'] and not r['chufang'])
        complete_cases = sum(1 for r in improved_results if r['zhenduan'] and r['bianzheng'] and r['chufang'])
        
        print("\n" + "=" * 60)
        print("处理完成！最终统计结果：")
        print("=" * 60)
        print(f"总案例数：{total}")
        print(f"有诊断：{zhenduan_count} ({zhenduan_count/total*100:.1f}%)")
        print(f"有辨证：{bianzheng_count} ({bianzheng_count/total*100:.1f}%)")
        print(f"有处方：{chufang_count} ({chufang_count/total*100:.1f}%)")
        print(f"三项俱全：{complete_cases} ({complete_cases/total*100:.1f}%)")
        print(f"完全为空：{all_empty} ({all_empty/total*100:.1f}%)")
        print("\n生成的文件：")
        print("- zhenduan.json: 从txt文件提取的案例诊断信息")
        print("- gold_standard.json: 真实的诊断、辨证、处方标准答案")
        print("- rag_output.json: RAG系统输出的诊断、辨证、处方结果（增量保存，支持中断恢复）")
        print("- rag_full_responses.json: RAG系统每个案例的完整回答内容（增量保存，支持中断恢复）")
        print("- rag_output_improved.json: 经过改进算法优化的最终结果")
    
    def clean_response(self, response: str, include_diagnostic_reasoning: bool = False) -> str:
        """清理模型输出，移除思考过程和多余内容，处理三行或四行输出格式"""
        if not response:
            return ""
        
        # 移除 <think>...</think> 标签及其内容
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 移除所有格式标记（**、*等）
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
        response = re.sub(r'\*([^*]+)\*', r'\1', response)
        
        # 按行分割并清理
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # 存储找到的各个字段
        diagnosis = ""
        syndrome = ""
        prescription = ""
        reasoning = ""
        
        # 逐行解析
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('诊断:') and not line.startswith('诊断理由:'):
                content = line.split(':', 1)[1].strip()
                if content:
                    diagnosis = content
            
            elif (line.startswith('辩证:') or line.startswith('辨证:')):
                content = line.split(':', 1)[1].strip()
                if content:
                    syndrome = content
                    
            elif line.startswith('处方:'):
                content = line.split(':', 1)[1].strip()
                # 查找后续的用法用量信息
                prescription_text = content
                j = i + 1
                while j < len(lines) and j < i + 5:  # 最多查看后5行
                    next_line = lines[j].strip()
                    if '【用法用量】' in next_line:
                        prescription_text += f"   {next_line}"
                        break
                    elif '用法' in next_line or '煎服' in next_line or ('共' in next_line and '剂' in next_line):
                        prescription_text += f"   【用法用量】{next_line}"
                        break
                    elif next_line and any(keyword in next_line for keyword in ['g', '克']):
                        prescription_text += f"   {next_line}"
                    elif next_line.startswith('诊断理由') or next_line.startswith('辩证') or next_line.startswith('诊断'):
                        break
                    j += 1
                
                # 如果没有找到用法用量，添加默认的
                if '【用法用量】' not in prescription_text:
                    prescription_text += "   【用法用量】共7剂，煎服1日2次1日1剂 每次200ml"
                
                # 去重处理
                prescription = self.remove_duplicate_herbs(prescription_text)
                
            elif line.startswith('诊断理由:') and include_diagnostic_reasoning:
                content = line.split(':', 1)[1].strip()
                # 可能诊断理由会跨多行，收集后续内容
                reason_content = content
                j = i + 1
                while j < len(lines) and j < i + 10:  # 最多查看后10行
                    next_line = lines[j].strip()
                    if next_line and not any(prefix in next_line for prefix in ['诊断:', '辩证:', '处方:', '诊断理由:']):
                        reason_content += f" {next_line}"
                    else:
                        break
                    j += 1
                reasoning = reason_content
                break
            
            i += 1
        
        # 构建结果
        result_lines = []
        
        if diagnosis:
            result_lines.append(f"诊断: {diagnosis}")
        else:
            result_lines.append("诊断: 未明确")
            
        if syndrome:
            result_lines.append(f"辩证: {syndrome}")
        else:
            result_lines.append("辩证: 未明确")
            
        if prescription:
            result_lines.append(f"处方: {prescription}")
        else:
            result_lines.append("处方: 未明确   【用法用量】请咨询医师")
            
        if include_diagnostic_reasoning:
            if reasoning:
                result_lines.append(f"诊断理由: {reasoning}")
            else:
                result_lines.append("诊断理由: 需要进一步分析")
        
        return '\n'.join(result_lines)
    
    def remove_duplicate_herbs(self, prescription: str) -> str:
        """去除处方中的重复药物"""
        if not prescription:
            return prescription
            
        # 分离处方和用法用量
        if '【用法用量】' in prescription:
            herbs_part, usage_part = prescription.split('【用法用量】', 1)
            usage_part = '【用法用量】' + usage_part
        else:
            herbs_part = prescription
            usage_part = ""
        
        # 解析药物
        import re
        herb_pattern = r'([^\s]+?)\s*(\d+g(?:先煎|后下|冲服|包煎)*)'
        herbs = re.findall(herb_pattern, herbs_part)
        
        # 去重，保留第一次出现的药物
        seen_herbs = set()
        unique_herbs = []
        for herb, dose in herbs:
            herb_clean = re.sub(r'(先煎|后下|冲服|包煎)', '', herb).strip()
            if herb_clean not in seen_herbs:
                seen_herbs.add(herb_clean)
                unique_herbs.append(f"{herb} {dose}")
        
        # 重新组合
        cleaned_prescription = '   '.join(unique_herbs)
        if usage_part:
            cleaned_prescription += '   ' + usage_part
            
        return cleaned_prescription


def main():
    """主函数"""
    # 可以根据需要修改这些参数
    processor = TCMRAGProcessor(
        knowledge_base_dir="./data",
        embedding_model_name="/home/luolihui/research/llm/models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        llm_model_name="qwen3:8b",
        faiss_index_path="faiss_index"
    )
    
    # 运行完整处理流程
    processor.run_complete_process()


if __name__ == "__main__":
    main()
