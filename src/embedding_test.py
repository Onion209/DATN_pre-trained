import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from data_processing import load_data_from_folder

VECTOR_DB_PATH = "vectorstores/db_faiss"
FOLDER_PATH = "./data"

# Validate và kiểm tra các document
def validate_inputs(documents, max_length=512):
    valid_docs = []
    for i, doc in enumerate(documents):
        if len(doc.strip()) == 0:
            print(f"Warning: Document {i} is empty, skipping...")
            continue
        tokens = doc.split()
        if len(tokens) > max_length:
            print(f"Warning: Document {i} is too long ({len(tokens)} tokens), truncating...")
            doc = " ".join(tokens[:max_length])
        valid_docs.append(doc)
    return valid_docs

# Tạo embeddings và lưu FAISS
def generate_embeddings_and_save(documents, model_name="dangvantuan/vietnamese-embedding"):
    # Khởi tạo mô hình embedding
    print("Loading embedding model...")
    model = HuggingFaceEmbeddings(model_name=model_name)

    # Validate documents
    documents = validate_inputs(documents)
    print(f"Validated {len(documents)} documents.")

    # Tạo và lưu FAISS database
    print("Generating embeddings and saving to FAISS...")
    try:
        db = FAISS.from_texts(documents, embedding=model)
        db.save_local(VECTOR_DB_PATH)
        print(f"FAISS database saved to {VECTOR_DB_PATH}.")
    except Exception as e:
        print(f"Error during FAISS save: {e}")

# Tải FAISS và thực hiện tìm kiếm
def load_and_search_faiss(query, model_name="dangvantuan/vietnamese-embedding", top_k=5):
    print("Loading FAISS database...")
    model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(VECTOR_DB_PATH, embedding=model, allow_dangerous_deserialization=True)

    # Thực hiện tìm kiếm
    print(f"Searching for: {query}")
    results = db.similarity_search(query, k=top_k)
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result.page_content}")

if __name__ == "__main__":
    # Bước 1: Tải dữ liệu
    print("Loading documents from folder...")
    documents, metadata = load_data_from_folder(FOLDER_PATH)
    print(f"Loaded {len(documents)} documents.")

    # Bước 2: Tạo embeddings và lưu FAISS
    generate_embeddings_and_save(documents, model_name="dangvantuan/vietnamese-embedding")

    # Bước 3: Tìm kiếm thử nghiệm
    test_query = "Thông tin tuyển sinh của Đại học Bách Khoa Hà Nội"
    load_and_search_faiss(test_query)
