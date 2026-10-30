# Linear Regression — bài 3

[Bài gốc](https://machinelearningcoban.com/2016/12/28/linearregression/). `generate.py` tạo hai CSV 100 hàng: bộ sạch và bản sao có outlier. Mỗi hàng có 10 đặc trưng; `target` và `is_outlier` không thuộc ma trận đặc trưng.

`feature_matrix` có shape `(100, 10)`. Bản Manual ghép cột 1 thành `(100, 11)`, tính pseudoinverse rồi nhân với `target_values` shape `(100,)` để lấy bias và 10 trọng số. Kết quả gồm MSE trên toàn bộ dữ liệu và MSE trên tập holdout từ một lần fit riêng.

Chạy: `uv run python -m data.script.linear_regression.generate` rồi `uv run python -m src.linear_regression.run`.
