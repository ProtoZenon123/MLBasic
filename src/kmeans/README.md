# K-means — bài 4–5

[Bài 4](https://machinelearningcoban.com/2017/01/01/kmeans/) và [bài 5](https://machinelearningcoban.com/2017/01/04/kmeans2/). Dữ liệu gồm 270 điểm 2 chiều có nhãn vùng chỉ để kiểm tra, cùng 576 pixel RGB của ảnh tự sinh.

Manual K-means lấy hiệu bằng broadcasting: `(n, 1, d) - (1, k, d)` tạo khoảng cách `(n, k)` sau khi cộng theo trục đặc trưng. Tâm cụm là trung bình các hàng được gán vào cụm. Phần ảnh dùng scikit-learn và báo lỗi tái tạo pixel.

Chạy: `uv run python -m data.script.kmeans.generate` rồi `uv run python -m src.kmeans.run`.
