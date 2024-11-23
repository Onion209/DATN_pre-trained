import os
from docx import Document
from PyPDF2 import PdfReader

# Load data
def read_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    return text
def split_data(text, chunk_size = 100, overlap = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks
# Duyệt qua toàn bộ thư mục data
def load_data_from_folder(folder_path):
    documents = []
    metadata = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.docx'):
                text = read_docx(file_path)
            elif file.endswith('.pdf'):
                text = read_pdf(file_path)
            else:
                continue
            chunks = split_data(text)
            # print(f"File {file} created {len(chunks)} chunks.")
            documents.extend(chunks)
            metadata.extend([{"file_name": file, "folder": root, "chunk": chunk} for chunk in chunks])
    # print(f"Total documents loaded: {len(documents)}")
    return documents, metadata

# # documents = load_document("data")
if __name__ == "__main__":
    folder_path = "./data"
    documents, metadata = load_data_from_folder(folder_path)  # Lấy cả documents và metadata
    # Đếm và in số lượng chunk đã tạo
    print(f"Total chunks created: {len(documents)}")

    #     # In thử chunk đầu tiên
    if documents:
        print(f"Chunk 1: {documents[366]}")
        # print(f"chunk 2: {documents[1]}")
        # print(f"chunk 3: {documents[2]}")
    else:
        print("No chunks loaded.")

