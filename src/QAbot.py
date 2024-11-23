from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

db_path = "/home/minhlahanhne/DATN_pre-trained/faiss_index.index"
model_path = "/home/minhlahanhne/DATN_pre-trained/model/vinallama-7b-chat_q5_0.gguf"
# Load model
def load_model(model_path):
    model = CTransformers(
        model = model_path,
        model_type = "llama",
        max_length = 512,
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
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)
llm_chain = create_bot_chain(prompt, model, db)

# run
question = """Các phương thức tuyển sinh ĐHCQ áp dụng cho đối tượng học sinh trung 
học phổ thông (THPT) nào? """
response = llm_chain.invoke({"query": question})
print(response)