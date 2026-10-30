# Gradient Descent — bài 7–8

[Phần 1](https://machinelearningcoban.com/2017/01/12/gradientdescent/) và [phần 2](https://machinelearningcoban.com/2017/01/16/gradientdescent2/) ở cùng thư mục. CSV hồi quy có 240 hàng, ba đặc trưng; ví dụ bậc hai có hai tọa độ và độ cong riêng.

Ma trận thiết kế thêm bias có shape `(n_train, 4)`. Gradient của half-MSE là `design.T @ (design @ weights - targets) / batch_size`, kết quả shape `(4,)`. `descend()` thay kích thước batch và quy tắc cập nhật để chạy batch, SGD, mini-batch, momentum, Nesterov; scikit-learn LinearRegression là nghiệm tham chiếu.

Chạy: `uv run python -m data.script.gradient_descent.generate` rồi `uv run python -m src.gradient_descent.run`.
