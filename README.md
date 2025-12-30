# Wine Quality Prediction using Machine Learning

---

## 1. Giới thiệu đề tài
- **Bài toán**: Dự đoán chất lượng rượu vang (Wine Quality Prediction) dựa trên các đặc trưng hóa học.
- **Loại bài toán**: Classification / Regression (tùy cách xây dựng mô hình).
- **Mục tiêu**:
  - Xây dựng pipeline Machine Learning hoàn chỉnh
  - Huấn luyện và đánh giá mô hình
  - Triển khai demo dự đoán bằng Streamlit

---

## 2. Dataset
- **Nguồn dữ liệu**:  
  UCI Machine Learning Repository – Wine Quality Dataset  
  Link: https://archive.ics.uci.edu/ml/datasets/wine+quality

- **Mô tả dữ liệu**:

| Cột | Ý nghĩa |
|---|---|
| fixed acidity | Độ axit cố định |
| volatile acidity | Độ axit bay hơi |
| citric acid | Axit citric |
| residual sugar | Lượng đường dư |
| chlorides | Hàm lượng muối |
| free sulfur dioxide | SO₂ tự do |
| total sulfur dioxide | Tổng SO₂ |
| density | Mật độ |
| pH | Độ pH |
| sulphates | Sulphates |
| alcohol | Nồng độ cồn |
| quality | Chất lượng rượu (nhãn) |

- **Số lượng mẫu**: ~1600
- **Chia tập dữ liệu**:
  - Train: 80%
  - Test: 20%

---

## 3. Pipeline xử lý
Quy trình xử lý dữ liệu và huấn luyện mô hình:

Dataset
↓
Preprocessing

Xử lý missing value

Chuẩn hóa dữ liệu (StandardScaler)
↓
Train Model
↓
Evaluate
↓
Inference / Streamlit Demo

yaml
Sao chép mã

---

## 4. Mô hình sử dụng
Các mô hình được áp dụng:
- Logistic Regression
- Decision Tree
- Random Forest

### Lý do lựa chọn:
- Logistic Regression: mô hình baseline, dễ hiểu
- Decision Tree: mô hình phi tuyến, dễ diễn giải
- Random Forest: cải thiện độ chính xác, giảm overfitting

---

## 5. Kết quả
### Đánh giá mô hình:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix

(Random Forest cho kết quả tốt nhất trong các mô hình thử nghiệm)

---

## 6. Hướng dẫn chạy dự án

### 6.1 Cài đặt môi trường
Khuyến nghị sử dụng **Python 3.9+**

```bash
pip install -r requirements.txt
```
---
### 6.2 Chạy ứng dụng Streamlit
Sau khi đã cài đặt đầy đủ các thư viện, bạn có thể khởi chạy giao diện demo bằng lệnh sau:

```bash
streamlit run app.py
```
