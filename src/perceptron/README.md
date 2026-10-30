# Perceptron — bài 9

[Bài gốc](https://machinelearningcoban.com/2017/01/21/perceptron/). Hai CSV có cùng 200 điểm hai chiều; bản noisy đổi 22 nhãn để quan sát giới hạn của mô hình tuyến tính.

Manual đổi nhãn `0/1` thành `-1/+1`. Với mỗi hàng shape `(2,)`, dấu của `row @ coef_ + intercept_` quyết định có cập nhật `coef_` và bias hay không. So test accuracy, số epoch và số lần cập nhật Manual với scikit-learn.

Chạy: `uv run python -m data.script.perceptron.generate` rồi `uv run python -m src.perceptron.run`.
