# Linear Discriminant Analysis — bài 29

[Bài gốc](https://machinelearningcoban.com/2017/06/30/lda/). CSV có 360 hàng, bốn đặc trưng và ba lớp.

Manual tính within-class và between-class scatter, mỗi ma trận `(4, 4)`, rồi chọn hai eigenvector Fisher thành phép chiếu `(4, 2)`. Test được gán theo tâm lớp sau chiếu. So shape chiếu và accuracy với scikit-learn LDA; hai bộ phân lớp không buộc có trọng số giống nhau.

Chạy: `uv run python -m data.script.lda.generate` rồi `uv run python -m src.lda.run`.
