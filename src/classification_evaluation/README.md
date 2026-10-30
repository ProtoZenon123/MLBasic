# Classification Evaluation — bài 33

[Bài gốc](https://machinelearningcoban.com/2017/08/31/evaluation/). Hai CSV có nhãn thật, nhãn dự đoán và score cho 180 mẫu nhị phân hoặc 180 mẫu ba lớp.

Manual dùng `np.add.at` để cộng vào confusion matrix `(classes, classes)`, rồi tính precision/recall/F1 micro và macro. ROC AUC nhị phân được tích phân sau khi gom cùng score thành một ngưỡng. Tất cả giá trị được đối chiếu với `sklearn.metrics`.

Chạy: `uv run python -m data.script.classification_evaluation.generate` rồi `uv run python -m src.classification_evaluation.run`.
