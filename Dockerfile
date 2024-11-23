# Sử dụng Python 3.9 làm image cơ bản
FROM python:3.11.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirement.txt .

# Cài đặt các dependencies từ requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Chỉ định cổng mà ứng dụng sẽ chạy (nếu cần)
EXPOSE 8000

# Lệnh mặc định để chạy ứng dụng (tuỳ chỉnh theo ứng dụng của bạn)
CMD ["python", "src/prepare_vectorDB.py"]
