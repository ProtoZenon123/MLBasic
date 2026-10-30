# MLBasic

Thực hành các bài có code trong [Machine Learning cơ bản](https://machinelearningcoban.com/categories/)
bằng dữ liệu tự sinh. Mỗi bài toán có một thư mục trong `src/` và
`data/script/`; bài gồm nhiều phần dùng chung một thư mục.

## Chạy một bài toán

```bash
uv run python -m data.script.<problem>.generate
uv run python -m src.<problem>.run
```

Thay `<problem>` bằng tên thư mục ở bảng dưới. Dữ liệu sinh nằm trong
`data/generated_data/<problem>/`, kết quả nằm trong `outputs/<problem>/results.txt`.
Hai thư mục này được Git bỏ qua. Các chương trình dùng seed cố định và so sánh
trên cùng dữ liệu; với bài học có bản tự cài đặt, kết quả ghi nhãn `Manual`.

| Bài | Thư mục | Nội dung |
| --- | --- | --- |
| 3 | `linear_regression` | Pseudoinverse, dữ liệu sạch và outlier |
| 4–5 | `kmeans` | Phân cụm và lượng tử hóa ảnh màu tự sinh |
| 6 | `knn` | Phân lớp theo láng giềng gần |
| 7–8 | `gradient_descent` | Batch, SGD, mini-batch, momentum, Nesterov |
| 9 | `perceptron` | Quy tắc cập nhật Perceptron |
| 10 | `logistic_regression` | Phân lớp nhị phân bằng sigmoid |
| 12 | `binary_classifiers` | Nhị phân, One-vs-Rest, One-vs-One |
| 13 | `softmax_regression` | Phân lớp nhiều lớp bằng softmax |
| 15 | `overfitting` | Bậc đa thức, validation, Ridge |
| 17 | `convex_optimization` | Geometric programming bằng CVXOPT |
| 19–22 | `svm` | Hard margin, soft margin, kernel, multiclass |
| 23 | `content_based_recommendation` | Item profile, TF-IDF, Ridge |
| 24 | `neighborhood_cf` | User-user và item-item CF |
| 25 | `matrix_factorization_cf` | Factorization với gradient và Surprise |
| 26 | `svd` | Phân rã và xấp xỉ hạng thấp |
| 27–28 | `pca` | Giảm chiều và quan hệ với SVD |
| 29 | `lda` | Chiếu Fisher và phân lớp |
| 32 | `naive_bayes` | Multinomial và Bernoulli Naive Bayes |
| 33 | `classification_evaluation` | Confusion, precision, recall, F1, ROC AUC |
| 34 | `id3` | Cây phân lớp ID3 và tham chiếu CART |

Các bài giới thiệu hoặc thuần lý thuyết 1–2, 11, 16, 18, 30–31 không có
script. Bài 14 và 35–37 dành cho repo Deep Learning. Mỗi thư mục `src/<problem>/`
có README riêng về shape dữ liệu và phép tính chính. Bản framework chỉ dùng
framework ở những bài blog không tự cài đặt mô hình.

## Linear regression

```bash
uv run python -m data.script.linear_regression.generate
uv run python -m src.linear_regression.run
```

Terminal chỉ hiển thị các bước đang chạy hoặc đã hoàn thành. Kết quả của bản
Manual và scikit-learn được ghi vào `outputs/linear_regression/results.txt` với
8 chữ số thập phân cho trọng số và MSE. Bài này không tạo ảnh; công cụ vẽ dùng
chung vẫn nằm ở `src/tools/plotting.py` cho các bài sau. `outputs/` và dữ liệu
sinh ra được Git bỏ qua.

Logger dùng chung cho các bài toán nằm ở `src/tools/logger.py`: gọi
`configure_logger(path_to_results_txt)` một lần, dùng `logger.info()` cho tiến
trình và `logger.log("RESULT", ...)` cho kết quả.

Kiểm tra các hành vi quan trọng bằng:

```bash
uv run python -m unittest discover -s tests -v
```
