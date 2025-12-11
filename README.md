# 🛍️ Customer Segmentation & Analysis Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-KMeans-yellow)

## 📖 Giới thiệu (Introduction)
Dự án này xây dựng hệ thống phân khúc khách hàng tự động sử dụng thuật toán **K-Means Clustering**. Mục tiêu là giúp doanh nghiệp/bộ phận Marketing hiểu rõ hơn về cơ sở khách hàng, từ đó đưa ra các chiến lược tiếp thị mục tiêu (Targeted Marketing) phù hợp cho từng nhóm, tối ưu hóa chi phí và gia tăng doanh thu.

Link Demo App: [Thêm link Streamlit của bạn vào đây sau khi deploy]

## 🎯 Mục tiêu Kinh doanh (Business Objectives)
- **Phân nhóm khách hàng:** Tự động chia khách hàng thành các nhóm dựa trên hành vi mua sắm (Thu nhập, Điểm chi tiêu, Tuổi tác).
- **Cá nhân hóa:** Đề xuất chiến lược chăm sóc riêng biệt cho nhóm khách hàng VIP, khách hàng tiềm năng, hoặc khách hàng vãng lai.
- **Trực quan hóa:** Cung cấp cái nhìn 3D trực quan về phân bố khách hàng.

## 🛠️ Công nghệ sử dụng (Tech Stack)
- **Ngôn ngữ:** Python
- **Xử lý dữ liệu:** Pandas, NumPy
- **Trực quan hóa:** Matplotlib, Seaborn, Plotly (cho biểu đồ 3D)
- **Machine Learning:** Scikit-learn (K-Means Clustering, Silhouette Score)
- **Deployment:** Streamlit Cloud

## 📊 Kết quả Phân tích (Key Insights)
*Mô hình đã phân chia khách hàng thành 5 nhóm chính:*
1.  **Nhóm Cẩn trọng (Tiết kiệm):** Thu nhập thấp, Chi tiêu thấp. -> *Chiến lược: Khuyến mãi giá rẻ.*
2.  **Nhóm Tiêu chuẩn:** Thu nhập trung bình, Chi tiêu trung bình. -> *Chiến lược: Giữ chân bằng CSKH chuẩn.*
3.  **Nhóm Mục tiêu (Tiềm năng):** Thu nhập cao, Chi tiêu thấp. -> *Chiến lược: Kích cầu mua sắm bằng sản phẩm cao cấp.*
4.  **Nhóm Phóng khoáng (Rủi ro):** Thu nhập thấp, Chi tiêu cao. -> *Chiến lược: Giới thiệu các gói trả góp, thẻ thành viên.*
5.  **Nhóm VIP:** Thu nhập cao, Chi tiêu cao. -> *Chiến lược: Dịch vụ đặc biệt, thẻ đen, ưu đãi độc quyền.*

## 🚀 Hướng dẫn cài đặt (Installation)

1. Clone repository:
```bash
git clone [https://github.com/nnn27072005/Customer_segmentation.git](https://github.com/nnn27072005/Customer_segmentation.git)
cd Customer_segmentation
