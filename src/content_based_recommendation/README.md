# Content-based Recommendation — bài 23

[Bài gốc](https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/). Generator tạo 42 item với sáu thể loại nhị phân và rating của 18 user; mỗi user giữ năm rating để test.

`TfidfTransformer` đổi ma trận item `(42, 6)` thành item profile cùng shape. Với từng user, `Ridge` học từ profile item đã đánh giá để dự đoán rating chưa thấy. Bài này dùng scikit-learn và báo MAE/RMSE trên rating test.

Chạy: `uv run python -m data.script.content_based_recommendation.generate` rồi `uv run python -m src.content_based_recommendation.run`.
