#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCM RAG Diagnosis System - Standalone Version

Features:
- Knowledge-based TCM intelligent diagnosis
- Support for tongue diagnosis, pulse diagnosis and other multi-dimensional information
- Structured output of diagnosis, syndrome differentiation, prescription and reasoning

Usage:
1. Ensure required dependencies are installed: pip install -r requirements.txt
2. Ensure Ollama service is running and qwen3:8b model is installed
3. Run: python tcm_rag_diagnosis.py

"""

import os
import re
import json
import time
from typing import Dict, Any, Optional
from operator import itemgetter

# LangChain相关导入
try:
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装依赖: pip install -r requirements.txt")
    exit(1)


class TCMRAGDiagnosis:
    """TCM RAG Diagnosis System"""
    
    def __init__(self, 
                 knowledge_base_dir="./knowledge_base",
                 embedding_model_name="BAAI/bge-small-zh-v1.5",
                 llm_model_name="qwen3:8b",
                 faiss_index_path="./faiss_index"):
        
        self.knowledge_base_dir = knowledge_base_dir
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.faiss_index_path = faiss_index_path
        
        # Initialize components
        self.vectorstore = None
        self.llm = None
        self.rag_chain = None
        
        # Prompt template
        self.prompt_template = """
你是一名中医专家，根据资料库中提供的资料以及患者信息（患者现状、舌诊、脉诊），给出诊断、辩证、处方和诊断理由。

严格按照以下格式输出，不要输出任何其他内容：

诊断: [病名]
辩证: [辩证结果]
处方: [药物1] [剂量]   [药物2] [剂量]   [药物3] [剂量]   【用法用量】[用法说明]
诊断理由: [详细的诊断分析和理由，包括症状分析、舌脉分析、病机阐述]

要求：
1. 必须输出上述四行，每行以"诊断:"、"辩证:"、"处方:"、"诊断理由:"开头
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
        print("📚 开始加载和处理知识库文档...")
        
        if not os.path.exists(directory_path):
            print(f"❌ 知识库目录不存在: {directory_path}")
            return None
            
        loader = DirectoryLoader(directory_path, glob="**/*.txt", show_progress=True)
        documents = loader.load()
        
        if not documents:
            print(f"❌ 在目录 '{directory_path}' 下未找到任何 .txt 文档")
            return None

        print(f"✅ 成功加载 {len(documents)} 个文档")
        
        # 文档切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", "。", "；", "，", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"✅ 文档切分完成，共 {len(split_docs)} 个文档块")
        
        return split_docs

    def create_and_save_vectorstore(self, docs, embedding_model_name: str, index_path: str):
        """创建并保存向量数据库"""
        print("🔄 创建向量数据库...")
        
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(index_path)
            print(f"✅ 向量数据库创建并保存到: {index_path}")
            return vectorstore
            
        except Exception as e:
            print(f"❌ 创建向量数据库时出错: {e}")
            return None

    def load_vectorstore(self, index_path: str, embedding_model_name: str):
        """加载已存在的向量数据库"""
        print("📖 加载向量数据库...")
        
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ 向量数据库加载成功")
            return vectorstore
            
        except Exception as e:
            print(f"❌ 加载向量数据库时出错: {e}")
            return None

    def initialize_system(self):
        """初始化RAG系统"""
        print("🚀 初始化中医RAG诊断系统...")
        
        # 检查FAISS索引是否存在
        if not os.path.exists(self.faiss_index_path):
            docs = self.load_and_process_documents(self.knowledge_base_dir)
            if docs:
                self.vectorstore = self.create_and_save_vectorstore(docs, self.embedding_model_name, self.faiss_index_path)
            else:
                print("❌ 未能处理任何文档，系统初始化失败")
                return False
        else:
            # 加载已存在的向量数据库
            self.vectorstore = self.load_vectorstore(self.faiss_index_path, self.embedding_model_name)
        
        if not self.vectorstore:
            print("❌ 向量数据库初始化失败")
            return False

        # 初始化Ollama LLM
        try:
            print(f"🤖 初始化LLM模型: {self.llm_model_name}")
            self.llm = Ollama(model=self.llm_model_name)
            
            # 测试模型连接
            test_response = self.llm.invoke("你好")
            print("✅ LLM模型连接成功")
            
        except Exception as e:
            print(f"❌ LLM模型初始化失败: {e}")
            print("请确保Ollama服务正在运行并已安装指定模型")
            return False
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
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
        
        print("✅ RAG系统初始化完成")
        return True

    def format_patient_info(self, patient_data: Dict[str, Any]) -> str:
        """格式化患者信息为查询文本"""
        query_parts = []
        
        if patient_data.get('zhusu'):
            query_parts.append(f"主诉：{patient_data['zhusu']}")
        
        if patient_data.get('xianbingshi'):
            query_parts.append(f"现病史：{patient_data['xianbingshi']}")
        
        if patient_data.get('shezhen'):
            query_parts.append(f"舌诊：{patient_data['shezhen']}")
        
        if patient_data.get('maizhen'):
            query_parts.append(f"脉诊：{patient_data['maizhen']}")
        
        return "\\n".join(query_parts)

    def extract_structured_info(self, response: str) -> Dict[str, str]:
        """从RAG响应中提取结构化信息"""
        result = {
            'zhenduan': '',
            'bianzheng': '',
            'chufang': '',
            'zhenduan_liyou': ''
        }
        
        try:
            lines = response.strip().split('\\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('诊断:'):
                    result['zhenduan'] = line.replace('诊断:', '').strip()
                elif line.startswith('辩证:'):
                    result['bianzheng'] = line.replace('辩证:', '').strip()
                elif line.startswith('处方:'):
                    result['chufang'] = line.replace('处方:', '').strip()
                elif line.startswith('诊断理由:'):
                    result['zhenduan_liyou'] = line.replace('诊断理由:', '').strip()
        
        except Exception as e:
            print(f"⚠️ 信息提取出错: {e}")
        
        return result

    def diagnose(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """进行中医诊断"""
        if not self.rag_chain:
            print("❌ RAG系统未初始化")
            return None
        
        try:
            # 格式化患者信息
            query = self.format_patient_info(patient_data)
            print(f"🔍 查询内容：\\n{query}")
            
            # 执行RAG查询
            print("🤔 正在分析诊断...")
            start_time = time.time()
            response = self.rag_chain.invoke({"question": query})
            end_time = time.time()
            
            print(f"⏱️ 诊断完成，耗时: {end_time - start_time:.2f}秒")
            print(f"🏥 原始诊断结果：\\n{response}")
            
            # 提取结构化信息
            structured_result = self.extract_structured_info(response)
            
            # 构建完整结果
            diagnosis_result = {
                'CaseID': patient_data.get('CaseID', f'case_{int(time.time())}'),
                'patient_info': patient_data,
                'raw_response': response,
                'structured_result': structured_result,
                'diagnosis_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return diagnosis_result
            
        except Exception as e:
            print(f"❌ 诊断过程出错: {e}")
            return None

    def batch_diagnose(self, patients_data: list) -> list:
        """批量诊断"""
        results = []
        total = len(patients_data)
        
        print(f"🏥 开始批量诊断，共 {total} 个案例")
        
        for i, patient_data in enumerate(patients_data, 1):
            print(f"\\n📋 处理案例 {i}/{total}: {patient_data.get('CaseID', 'Unknown')}")
            
            result = self.diagnose(patient_data)
            if result:
                results.append(result)
                print(f"✅ 案例 {i} 诊断完成")
            else:
                print(f"❌ 案例 {i} 诊断失败")
        
        print(f"\\n🎉 批量诊断完成！成功诊断 {len(results)}/{total} 个案例")
        return results

    def save_results(self, results: list, filename: str = "diagnosis_results.json"):
        """保存诊断结果"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 诊断结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

    def print_diagnosis_summary(self, result: Dict[str, Any]):
        """打印诊断摘要"""
        if not result:
            return
        
        structured = result.get('structured_result', {})
        patient = result.get('patient_info', {})
        
        print("\\n" + "="*60)
        print(f"📋 案例ID: {result.get('CaseID', 'N/A')}")
        print(f"⏰ 诊断时间: {result.get('diagnosis_time', 'N/A')}")
        print("-"*60)
        print(f"👤 患者信息:")
        print(f"   主诉: {patient.get('zhusu', 'N/A')}")
        print(f"   舌诊: {patient.get('shezhen', 'N/A')}")
        print(f"   脉诊: {patient.get('maizhen', 'N/A')}")
        print("-"*60)
        print(f"🏥 诊断结果:")
        print(f"   诊断: {structured.get('zhenduan', 'N/A')}")
        print(f"   辩证: {structured.get('bianzheng', 'N/A')}")
        print(f"   处方: {structured.get('chufang', 'N/A')}")
        print(f"   理由: {structured.get('zhenduan_liyou', 'N/A')}")
        print("="*60)


def main():
    """主函数"""
    print("🏥 中医RAG诊断系统 v1.0")
    print("="*60)
    
    # 创建诊断系统实例
    diagnosis_system = TCMRAGDiagnosis()
    
    # 初始化系统
    if not diagnosis_system.initialize_system():
        print("❌ 系统初始化失败")
        return
    
    # 示例患者数据
    sample_patients = [
        {
            "CaseID": "demo_case_1",
            "shezhen": "舌红苔黄",
            "maizhen": "脉数有力",
            "zhusu": "咳嗽痰黄3天",
            "xianbingshi": "患者3天前受风寒后出现咳嗽，痰黄粘稠，伴有发热，口干喜饮"
        },
        {
            "CaseID": "demo_case_2", 
            "shezhen": "舌淡苔白",
            "maizhen": "脉细弱",
            "zhusu": "乏力失眠1月",
            "xianbingshi": "患者近1月来精神疲倦，夜寐不安，食欲不振，大便溏薄"
        }
    ]
    
    print("\\n🧪 使用示例数据进行测试诊断...")
    
    # 单个案例诊断演示
    print("\\n📋 单案例诊断演示:")
    result = diagnosis_system.diagnose(sample_patients[0])
    if result:
        diagnosis_system.print_diagnosis_summary(result)
    
    # 批量诊断演示
    print("\\n📋 批量诊断演示:")
    batch_results = diagnosis_system.batch_diagnose(sample_patients)
    
    # 保存结果
    if batch_results:
        diagnosis_system.save_results(batch_results, "demo_diagnosis_results.json")
    
    print("\\n✨ 演示完成！")
    print("\\n📖 使用说明:")
    print("1. 修改sample_patients列表添加您的患者数据")
    print("2. 调用diagnosis_system.diagnose(patient_data)进行单例诊断") 
    print("3. 调用diagnosis_system.batch_diagnose(patients_list)进行批量诊断")
    print("4. 使用diagnosis_system.save_results()保存结果")


if __name__ == "__main__":
    main()
