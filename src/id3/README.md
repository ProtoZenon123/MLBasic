# Decision Tree ID3 — bài 34

[Bài gốc](https://machinelearningcoban.com/2018/01/14/id3/). CSV có 360 quan sát, bốn cột phân loại và nhãn `target` từ một quy tắc thời tiết tự sinh.

Manual tính entropy, information gain, rồi đệ quy chia theo giá trị thuộc tính. Mỗi node lưu nhãn đa số để dự đoán khi gặp category chưa thấy. scikit-learn nhận one-hot encoding và dùng CART làm mô hình tham chiếu; CART không phải ID3 nên chỉ so chất lượng dự đoán, không so cấu trúc cây.

Chạy: `uv run python -m data.script.id3.generate` rồi `uv run python -m src.id3.run`.
