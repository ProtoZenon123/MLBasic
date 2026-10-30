# Overfitting — bài 15

[Bài gốc](https://machinelearningcoban.com/2017/03/04/overfitting/). CSV có 120 điểm một chiều từ đa thức bậc ba cộng nhiễu; không dùng ví dụ MLP thuộc repo DL.

Scikit-learn mở rộng `(n, 1)` thành các cột lũy thừa bằng `PolynomialFeatures`, rồi fit LinearRegression/Ridge. Báo MSE train, validation, test theo bậc; CV chỉ dùng dữ liệu trước test để chọn bậc.

Chạy: `uv run python -m data.script.overfitting.generate` rồi `uv run python -m src.overfitting.run`.
