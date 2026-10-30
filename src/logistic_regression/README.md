# Logistic Regression — bài 10

[Bài gốc](https://machinelearningcoban.com/2017/01/27/logisticregression/). Dữ liệu có 500 hàng, ba đặc trưng và nhãn nhị phân sinh từ xác suất sigmoid.

Manual thêm bias vào ma trận `(n, 4)`, tính vector logit `(n,)`, áp sigmoid và gradient `(4,)` của log loss. `predict_proba()` trả ma trận `(n, 2)` theo thứ tự lớp 0, 1. Đối chiếu scikit-learn không regularization.

Chạy: `uv run python -m data.script.logistic_regression.generate` rồi `uv run python -m src.logistic_regression.run`.
