# Principal Component Analysis — bài 27–28

[Phần 1](https://machinelearningcoban.com/2017/06/15/pca/) và [phần 2](https://machinelearningcoban.com/2017/06/21/pca2/) cùng thư mục. CSV có 300 hàng, sáu đặc trưng sinh từ hai biến ẩn.

Scikit-learn PCA chiếu `(n, 6)` thành `(n, k)` rồi `inverse_transform` về `(n, 6)`. So tỉ lệ phương sai giải thích và MSE tái tạo trên test theo `k`; singular values của PCA được đối chiếu với NumPy SVD trên train đã trừ trung bình.

Chạy: `uv run python -m data.script.pca.generate` rồi `uv run python -m src.pca.run`.
