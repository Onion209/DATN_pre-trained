from sentence_transformers import SentenceTransformer
from data_processing import load_data_from_folder
import os
import numpy as np
from pathlib import Path
import faiss

vector_db_path = "vectorstores/db_faiss"
folder_path = "vectordb"
def validate_inputs(documents, max_length=512):
    valid_docs = []
    for i, doc in enumerate(documents):
        if len(doc.strip()) == 0:
            #print(f"Warning: Document {i} is empty, skipping...")
            continue
        tokens = doc.split()
        if len(tokens) > max_length:
            #print(f"Warning: Document {i} is too long ({len(tokens)} tokens), truncating...")
            doc = " ".join(tokens[:max_length])
        valid_docs.append(doc)
        #print(f"Document {i}: {repr(doc[:100])}...")  # Hiển thị 100 ký tự đầu
    return valid_docs

# Sinh embeddings từ các tài liệu
def generate_embeddings(documents, model_name="dangvantuan/vietnamese-embedding"):
    # Khởi tạo mô hình
    model = SentenceTransformer(model_name, device="cpu")  # Sử dụng CPU để debug
    embeddings = []
    for i, doc in enumerate(documents):
        try:
            #print(f"Processing document {i}: {repr(doc[:100])}...")  # Hiển thị 100 ký tự đầu
            embedding = model.encode([doc], show_progress_bar=False)
            embeddings.append(embedding[0])
        except Exception as e:
            print(f"Error in batch {i}: {e}")
    return embeddings
# Hàm lưu embeddings vào file
def save_embeddings_to_folder(embeddings, metadata, folder_path="vectordb"):
    # Tạo folder nếu chưa tồn tại
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
        # Tạo tên file cho từng chunk
        file_name = f"{meta['file_name']}_chunk_{i}.npy"
        file_path = os.path.join(folder_path, file_name)
        
        # Lưu embedding dưới dạng .npy
        np.save(file_path, embedding)
        
        # Lưu metadata kèm file
        meta_file = file_path.replace(".npy", ".json")
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(str(meta))
    
    print(f"Saved {len(embeddings)} embeddings and metadata to '{folder_path}'.")
# Load embeddings từ folder vectordb
def load_embeddings(folder_path="vectordb"):
    embeddings = []
    metadata = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            # Load embedding
            embedding = np.load(os.path.join(folder_path, file_name))
            embeddings.append(embedding)
            
            # Metadata tương ứng
            meta_file = file_name.replace(".npy", ".json")
            metadata.append(meta_file)
    
    embeddings = np.array(embeddings).astype("float32")
    print(f"Loaded {len(embeddings)} embeddings from '{folder_path}'.")
    return embeddings, metadata
# Tạo FAISS Index
def create_faiss_index(embeddings, index_file="faiss_index.index"):
    # Xác định số chiều của vector
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index for {len(embeddings)} embeddings with dimension {dimension}.")

    # Tạo index FAISS (Flat L2)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Thêm embeddings vào index

    # Lưu index vào file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to '{index_file}'.")

# Load FAISS Index
def load_faiss_index(index_file="faiss_index.index"):
    index = faiss.read_index(index_file)
    print(f"FAISS index loaded from '{index_file}'.")
    return index

# Tìm kiếm trên FAISS Index
def search_faiss_index(index, query_vector, top_k=5):
    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)  # Định dạng query
    distances, indices = index.search(query_vector, top_k)
    return distances, indices

def search_faiss_index(index, query_text, model, k=5):
    # Mã hóa chuỗi truy vấn thành vector
    query_vector = model.encode([query_text], show_progress_bar=False)
    query_vector = np.array(query_vector).astype("float32")

    # Tìm kiếm trong FAISS
    distances, indices = index.search(query_vector, k)
    return distances, indices
if __name__ == "__main__":
    # 1. Load embeddings và metadata
    model_name = "dangvantuan/vietnamese-embedding"
    model = SentenceTransformer(model_name, device="cpu")
    folder_path = "vectordb"
    embeddings, metadata = load_embeddings(folder_path)

    # 2. Tạo FAISS index và lưu vào file
    index_file = "faiss_index.index"
    create_faiss_index(embeddings, index_file)
    index = faiss.read_index(index_file)
    # Query text
    query_text = "Điểm chuẩn xét tuyển tài năng 2024 ngành IT1"

    # Tìm kiếm
    distances, indices = search_faiss_index(index, query_text, model, k=5)

    # Hiển thị kết quả
    print("Top 5 kết quả tương tự:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"Rank {i}: Index {idx}, Distance {dist}")