# Softmax Regression — bài 13

[Bài gốc](https://machinelearningcoban.com/2017/02/17/softmax/). CSV có 420 hàng, hai đặc trưng và ba lớp chồng lấn.

Manual ghép bias thành ma trận `(n, 3)`, nhân với trọng số `(3, 3)` để được score `(n, 3)`. Trừ max theo từng hàng trước `exp` để ổn định số; gradient là `design.T @ (probabilities - one_hot) / n`. So cross-entropy và accuracy với scikit-learn.

Chạy: `uv run python -m data.script.softmax_regression.generate` rồi `uv run python -m src.softmax_regression.run`.
