from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from embedding_generator import create_faiss_index, search_faiss_index
from embedding_generator import load_embeddings
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

model_path ="/home/minhlahanhne/DATN_pre-trained/model/vinallama-7b-chat_q5_0.gguf"
index_file = "faiss_index.index"
folder_path = "vectordb"
def load_model(model_path):
    model = CTransformers(
        model = model_path,
        model_type = "llama",
        max_length = 512,
        temperature = 0.2,
    )
    return model
def preprocess_query(question):
    query = question.strip().lower()
    return query
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["context", "question"]) # context craw tu DB
    return prompt 
def main():
    model_name = "dangvantuan/vietnamese-embedding"
    model = SentenceTransformer(model_name, device="cpu")
    embeddings, metadata = load_embeddings(folder_path)
    question = input("Question:")
    create_faiss_index(embeddings, index_file)
    index = faiss.read_index(index_file)
    distances, indices = search_faiss_index(index, question, model, k=5)
    # Lấy metadata từ các chỉ mục (indices)
    retrieved_contexts = [metadata[idx] for idx in indices[0]]
    context = "\n".join(retrieved_contexts)
    # Xây dựng prompt cho mô hình Vinallama
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = create_prompt(template)
    formatted_prompt = prompt.format(context=context, question=question)
    # Khởi tạo mô hình Vinallama (Llama)
    llama = load_model(model_path)

    # Tạo câu trả lời từ mô hình Vinallama
    response = llama(formatted_prompt) 
    print("Trả lời:", response)

# Chạy chương trình
main()
