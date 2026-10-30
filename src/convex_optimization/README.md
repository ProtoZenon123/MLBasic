# Convex Optimization — bài 17

[Bài gốc](https://machinelearningcoban.com/2017/03/19/convexopt/). Generator tạo hai hệ số dương và ngân sách cho bài toán geometric programming khả thi.

CVXOPT giải trong tọa độ `(log x, log y)` shape `(2,)`, với mục tiêu `a/x + b/y` và ràng buộc `x + y <= budget`. Kết quả ghi nghiệm, giá trị mục tiêu và phần vi phạm ràng buộc; không gán nhãn Manual cho solver thư viện.

Chạy: `uv run python -m data.script.convex_optimization.generate` rồi `uv run python -m src.convex_optimization.run`.
