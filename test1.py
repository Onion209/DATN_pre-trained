from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
# Khai bao bien
data_path = "data"
vector_db_path = "vectorstores/db_faiss"
doc_data_path = "/home/minhlahanhne/DATN_pre-trained/data_test/1. thong_tin_chung"
pdf_data_path = "/home/minhlahanhne/DATN_pre-trained/data_test/2. de_an_ts"
def create_db_from_files():
    # Khai bao thu muc loader de quet thu muc 
    loader_pdf = DirectoryLoader(pdf_data_path, glob = "*.pdf", loader_cls = PyPDFLoader)
    loader_doc = DirectoryLoader(doc_data_path, glob = "*.docx", loader_cls = Docx2txtLoader)
    documents_pdf = loader_pdf.load()
    documents_doc = loader_doc.load()
    all_documents = documents_pdf + documents_doc 
    # print(all_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )
    chunks = text_splitter.split_documents(all_documents)
        # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db     
create_db_from_files()