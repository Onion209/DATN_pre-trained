from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from data_processing import load_data_from_folder

# Embedding data
def validate_inputs(documents):
    valid_docs = []
    for doc in documents:
        if len(doc.strip()) == 0:
            print("Warning: Empty chunk detected, skipping...")
            continue
        if len(doc.split()) > 512:
            print("Warning: Chunk too long, truncating...")
            doc = " ".join(doc.split()[:512])
        valid_docs.append(doc)
    return valid_docs
def generate_embeddings(documents, model_name="dangvantuan/vietnamese-embedding"):
    # Load model and tokenizer
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize documents as plain text (không trả về tensor)
    tokenized_documents = [tokenizer.decode(tokenizer.encode(doc, truncation=True, max_length=512)) for doc in documents]

    # Tạo embeddings trực tiếp từ danh sách chuỗi
    embeddings = model.encode(tokenized_documents, show_progress_bar=True, batch_size=8)
    return embeddings

if __name__ == "__main__":
    folder_path = "./data"

    # Load and split data
    documents, metadata = load_data_from_folder(folder_path)
    documents = validate_inputs(documents)  # Validate input
    print(f"Loaded {len(documents)} valid chunks from data.")

    # Generate embeddings
    embeddings = generate_embeddings(documents)
    print(f"Generated {len(embeddings)} embeddings.")
