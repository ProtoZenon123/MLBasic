# Matrix Factorization CF — bài 25

[Bài gốc](https://machinelearningcoban.com/2017/05/31/matrixfactorization/). Dùng rating thưa tự sinh của 24 user và 36 item, với rating test giữ riêng.

Manual giữ ma trận user factor `(24, 4)` và item factor `(36, 4)` cùng bias; tích vô hướng của hai vector bốn chiều cho phần tương tác. SGD chỉ cập nhật trên rating train. So MSE train và MAE/RMSE test với Surprise `SVD` trên cùng split.

Chạy: `uv run python -m data.script.matrix_factorization_cf.generate` rồi `uv run python -m src.matrix_factorization_cf.run`.
