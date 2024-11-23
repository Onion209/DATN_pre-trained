from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Tập dữ liệu nhỏ để kiểm tra
test_documents = ["Quy định xét tuyển tài năng năm 2024 Thời gian: 10-03-2024 Ngày 10/3/2024, Đại học Bách khoa Hà Nội đã công bố quy định về phương thức xét tuyển tài năng đối với tuyển sinh Đại học hệ chính quy năm 2024",
                  "anh sách Tổ tư vấn chuyên sâu các chương trình/ngành học Thời gian: 16-04-2023 STT 01, Mã & Tên chương trình: (BF1) Kỹ thuật Sinh học, Họ và tên: PGS. Nguyễn Lan Hương, Email: huong.nguyenlan@hust.edu.vn, Điện thoại: 0903247172 STT 02, Mã & Tên chương trình: (BF2) Kỹ thuật Thực phẩm, Họ và tên: TS. Nguyễn Thị Hạnh, Email: hanh.nguyenthi@hust.edu.vn, Điện thoại: 0985899457 STT 03, Mã & Tên chương trình: (BF-E12) Kỹ thuật Thực phẩm (CT tiên tiến), Họ và tên: PGS. Vũ Thu Trang, Email: trang.vuthu@hust.edu.vn, Điện thoại: 0934668283 STT 04, Mã & Tên chương trình: (BF-E19) Kỹ thuật Sinh"]


# Kiểm tra dữ liệu đầu vào
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
        print(f"Document {i}: {repr(doc[:100])}...")  # Hiển thị 100 ký tự đầu
    return valid_docs

# Sinh embeddings từ các tài liệu
def generate_embeddings(documents, model_name="dangvantuan/vietnamese-embedding"):
    # Khởi tạo mô hình
    model = SentenceTransformer(model_name, device="cpu")  # Sử dụng CPU để debug
    embeddings = []
    for i, doc in enumerate(documents):
        try:
            print(f"Processing document {i}: {repr(doc[:100])}...")  # Hiển thị 100 ký tự đầu
            embedding = model.encode([doc], show_progress_bar=False)
            embeddings.append(embedding[0])
        except Exception as e:
            print(f"Error in batch {i}: {e}")
    return embeddings

if __name__ == "__main__":
    # Xác thực dữ liệu đầu vào
    valid_documents = validate_inputs(test_documents)
    print(f"Validated {len(valid_documents)} documents.")

    # Sinh embeddings
    embeddings = generate_embeddings(valid_documents)
    print(f"Generated {len(embeddings)} embeddings.")

    # Hiển thị một số embeddings
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i}: {embedding[:10]}...")  # In 10 giá trị đầu
