    # Kiểm tra xem thư mục có dữ liệu không
    if not documents:
        print("No documents found in the folder!")
        return
    # Chia nhỏ dữ liệu thành chunks và lưu embedding vectors vào FAISS
    split_data(documents)