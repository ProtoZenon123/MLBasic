# Neighborhood Collaborative Filtering — bài 24

[Bài gốc](https://machinelearningcoban.com/2017/05/24/collaborativefiltering/). Dữ liệu rating thưa gồm 24 user, 36 item; năm rating mỗi user được giữ lại làm test.

Manual tạo ma trận user–item `(24, 36)` với `NaN` cho ô thiếu, trừ trung bình theo user hoặc item, rồi tính cosine similarity. Dự đoán là trung bình cộng phần sai lệch rating có trọng số của mười hàng xóm; so MAE/RMSE với Surprise `KNNWithMeans` cho cả user-user và item-item.

Chạy: `uv run python -m data.script.neighborhood_cf.generate` rồi `uv run python -m src.neighborhood_cf.run`.
