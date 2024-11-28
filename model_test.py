from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import re
db_path = "vectorstores/db_faiss"
model_path = "model/vinallama-7b-chat_q5_0.gguf"
# Load model
def load_model(model_path):
    model = CTransformers(
        model = model_path,
        model_type = "llama",
        max_length = 1024,
        temperature = 0.2,
    )
    return model
# Create prompt template
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables = ["context", "question"]) # context craw tu DB
    return prompt 
def create_bot_chain(prompt, model, db):
    bot_chain = RetrievalQA.from_chain_type(
        llm = model,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k": 3}),
        return_source_documents = False,
        chain_type_kwargs = {'prompt': prompt}  
    )
    return bot_chain
# Read from DB
def read_vector_db():
    #EmbeddingDB
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

#Run chain
db = read_vector_db()
model = load_model(model_path)
# Template
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)
llm_chain = create_bot_chain(prompt, model, db)

# run
question = """Email của PGS.TS. Vũ Hoàng Phương là gì?"""
response = llm_chain.invoke({"query": question})
# Trích xuất giá trị từ key 'result' trong dictionary
result_text = response['result']

# Sử dụng biểu thức chính quy để lấy phần cần thiết từ 'result'
# extracted_result = re.search(r"result': '(.*?)<\|im_end\|>", result_text)
print(result_text)