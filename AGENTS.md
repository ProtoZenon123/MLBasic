# Quy ước làm việc cho repo MLBasic

## Mục tiêu và phạm vi

- Repo này dùng để ôn tập Machine Learning cơ bản theo https://machinelearningcoban.com/ bằng cách tự cài đặt thuật toán, rồi giải cùng ví dụ bằng framework để đối chiếu.
- Giữ Deep Learning ở repo riêng. Các bài 14 (MLP và backpropagation), 35 (lược sử DL), 36 (Keras) và 37 (tích chập hai chiều) thuộc repo DL. Perceptron, softmax regression, gradient descent và overfitting vẫn là kiến thức nền trong repo ML.
- Người dùng đang học cách code và muốn hiểu NumPy. Khi được hỏi về code, giải thích rõ shape của mảng, phép toán vector hóa, ý nghĩa các method và dữ liệu đi qua từng bước.

## Cấu trúc và dữ liệu

- Mỗi bài toán có script sinh dữ liệu ở `data/script/<problem>.py` và phần thực hành thuật toán ở `src/<problem>.py`. Đặt các hàm thật sự dùng chung trong `src/tools/`; tránh gom mọi thứ vào một `utils.py`.
- Đối với dữ liệu, chỉ commit script sinh dữ liệu. Dữ liệu sinh ra nằm trong `data/generated_data/` và được Git bỏ qua. Dùng seed để chạy lại cho cùng kết quả; khi cần kiểm tra điểm yếu của thuật toán, tạo các bộ dữ liệu có thể so sánh trực tiếp, như bộ sạch và bộ có outlier.
- Đầu ra của từng bài toán nằm chung trong `outputs/<problem>/`, chẳng hạn `results.txt` và ảnh nếu bài đó cần ảnh. `outputs/` được Git bỏ qua. Không tạo thư mục `plots/` dùng chung cho nhiều bài.
- Code đọc dữ liệu trong `src/` phải lấy số lượng và tên đặc trưng từ file đã sinh, không cố định `feat_1` đến `feat_10`. Với CSV hiện tại, `target` là cột cần dự đoán và `is_outlier` là metadata, không đưa hai cột này vào ma trận đặc trưng.

## Cách viết thuật toán và ví dụ

- Ưu tiên code ngắn, dễ đọc, tên biến nói rõ vai trò, ví dụ `feature_matrix`, `target_values`, `feature_names`, `predictions`. Tên cột dữ liệu có thể là `feat_1`, `feat_2`, ...; tránh dùng `X`, `y` trong code mới khi tên đầy đủ giúp dễ hiểu hơn.
- Bản tự cài đặt dùng NumPy cho phép tính, pandas cho dữ liệu, matplotlib khi cần vẽ. Không dùng estimator của scikit-learn bên trong thuật toán tự cài đặt. Trong kết quả, gọi bản này là `Manual`; dùng tên framework cho bản đối chiếu, ví dụ `scikit-learn`.
- Class nhỏ với `.fit()` và `.predict()` phù hợp khi mô hình cần lưu trọng số. Dùng hàm đơn giản cho công cụ không cần trạng thái; `measure_time()` hiện tại không cần chuyển thành class.
- So sánh Manual và framework trên cùng dữ liệu, báo các tham số và chỉ số phù hợp. Với Linear Regression, báo bias, trọng số, MSE và thời gian fit/predict. Các hệ số và MSE hiện được ghi với 8 chữ số thập phân.
- Chỉ vẽ khi hình giúp hiểu bài toán. Linear Regression hiện có 10 đặc trưng nên không tạo hình; giữ `src/tools/plotting.py` để dùng cho bài khác.

## Log, phụ thuộc và kiểm tra

- Dùng Loguru cho checkpoint tiến trình trên terminal: bắt đầu, đang xử lý, hoàn thành. Ghi kết quả chi tiết vào `outputs/<problem>/results.txt`, không in trọng số hay metric lên terminal. Tái sử dụng `src/tools/logger.py`: `configure_logger(...)`, `logger.info(...)` và `logger.log("RESULT", ...)`.
- Dùng `uv add <tên-package>` để thêm phụ thuộc và cập nhật `pyproject.toml`, `uv.lock`. Chạy script bằng `uv run python ...`.
- Sau thay đổi, chạy luồng liên quan từ sinh dữ liệu đến tính kết quả; kiểm tra file đầu ra và việc Git bỏ qua dữ liệu/đầu ra. Viết test khi có rủi ro hành vi đáng kiểm tra, tránh test chỉ lặp lại code triển khai.
- Giữ README và hướng dẫn chạy đồng bộ với source. Tôn trọng các chỉnh sửa người dùng đã làm trong repo, nhất là tên biến, nhãn kết quả và tham số dữ liệu; không tự ý thay chúng khi làm việc khác.
