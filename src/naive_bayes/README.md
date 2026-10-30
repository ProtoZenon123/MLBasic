# Naive Bayes — bài 32

[Bài gốc](https://machinelearningcoban.com/2017/08/08/nbc/). CSV có 360 hàng, sáu cột đếm từ và ba lớp. Cùng split dùng cho hai mô hình scikit-learn.

`MultinomialNB` nhận ma trận số đếm `(n, 6)`; `BernoulliNB` nhận ma trận nhị phân cùng shape từ phép `counts > 0`. Báo test accuracy và confusion matrix. Bài này không tạo bản Manual vì ví dụ code của blog dùng framework.

Chạy: `uv run python -m data.script.naive_bayes.generate` rồi `uv run python -m src.naive_bayes.run`.
