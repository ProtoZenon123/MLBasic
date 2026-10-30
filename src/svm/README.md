# Support Vector Machine — bài 19–22

[Bài 19](https://machinelearningcoban.com/2017/04/09/smv/), [20](https://machinelearningcoban.com/2017/04/13/softmarginsmv/), [21](https://machinelearningcoban.com/2017/04/22/kernelsmv/), [22](https://machinelearningcoban.com/2017/04/28/multiclasssmv/) cùng thư mục. Bốn CSV lần lượt có dữ liệu tách tuyến tính, chồng lấn, phi tuyến và ba lớp.

Hard/soft margin tạo ma trận Gram `(n_train, n_train)` và gửi bài toán đối ngẫu tới CVXOPT; so với `SVC` trên cùng split. Kernel chỉ dùng `SVC` RBF. Manual multiclass dùng score `(n, classes)`, hinge margin và gradient vector hóa cho trọng số `(features + bias, classes)`; so `LinearSVC`.

Chạy: `uv run python -m data.script.svm.generate` rồi `uv run python -m src.svm.run`.
