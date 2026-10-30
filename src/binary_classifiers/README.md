# Binary Classifiers — bài 12

[Bài gốc](https://machinelearningcoban.com/2017/02/11/binaryclassifiers/). CSV có 330 hàng hai chiều và ba lớp; ví dụ nhị phân dùng hai lớp đầu, ví dụ đa lớp dùng cả ba.

Bài này chỉ dùng scikit-learn: LogisticRegression nhị phân, `OneVsRestClassifier` và `OneVsOneClassifier`. Một lần chia train/test cho mỗi trường hợp, rồi so accuracy và confusion matrix.

Chạy: `uv run python -m data.script.binary_classifiers.generate` rồi `uv run python -m src.binary_classifiers.run`.
