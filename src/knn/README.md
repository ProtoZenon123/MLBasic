# K-nearest neighbors — bài 6

[Bài gốc](https://machinelearningcoban.com/2017/01/08/knn/). CSV có 300 hàng, hai đặc trưng và nhãn `target` gồm ba lớp; train/test được chia có stratify.

Trong Manual, phép trừ `(n_test, 1, 2) - (1, n_train, 2)` cho ma trận khoảng cách `(n_test, n_train)`. Mỗi hàng chọn năm điểm gần nhất rồi bỏ phiếu; hòa phiếu chọn nhãn nhỏ hơn.

Chạy: `uv run python -m data.script.knn.generate` rồi `uv run python -m src.knn.run`.
