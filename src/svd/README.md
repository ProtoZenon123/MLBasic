# Singular Value Decomposition — bài 26

[Bài gốc](https://machinelearningcoban.com/2017/06/07/svd/). File NPZ chứa ma trận `(40, 25)` tạo từ tích hạng ba cộng nhiễu nhỏ.

`numpy.linalg.svd(full_matrices=False)` trả `U (40, 25)`, singular values `(25,)` và `Vt (25, 25)`. Phép `(U * singular_values) @ Vt` tái tạo ma trận; khi chỉ lấy vài cột/hàng đầu sẽ được xấp xỉ hạng thấp. Báo sai số trực giao và Frobenius.

Chạy: `uv run python -m data.script.svd.generate` rồi `uv run python -m src.svd.run`.
