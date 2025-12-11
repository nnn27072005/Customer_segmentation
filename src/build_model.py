# File: build_model.py
import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import thư viện của bạn
sys.path.append('.')
from cluster_lib import ClusteringModel

print("🚀 Bắt đầu tạo lại model tương thích với máy local...")

# 1. Cấu hình đường dẫn (Sửa lại nếu file csv của bạn nằm chỗ khác)
scaled_path = '../data/processed/customer_features_scaled.csv'
original_path = '../data/processed/customer_features.csv'

# Kiểm tra file
if not os.path.exists(scaled_path):
    print(f"❌ Lỗi: Không tìm thấy file {scaled_path}")
    print("Hãy chạy notebook 01 và 02 để tạo dữ liệu trước!")
    sys.exit(1)

# 2. Load dữ liệu
analyzer = ClusteringModel(scaled_path, original_path)
df_scaled, df_original = analyzer.load_data()

# 3. Xử lý dữ liệu (Log -> Scale -> PCA)
print("⚙️ Đang xử lý dữ liệu...")
# Dùng Log Transform thay vì BoxCox để ổn định
X_log = np.log1p(df_original.select_dtypes(include=[np.number])) 

# Fit Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Fit PCA
pca = PCA(n_components=10) # Lấy 10 thành phần chính
X_pca = pca.fit_transform(X_scaled)

# 4. Train KMeans (K=4)
print("🧠 Đang train model KMeans (K=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_pca)

# 5. Lưu model
cluster_desc = {
    0: "Nhóm Premium (Chi tiêu cao)",
    1: "Nhóm Tiềm năng (Mới/Nhỏ)",
    2: "Nhóm Trung thành (Thường xuyên)",
    3: "Nhóm Vãng lai (Rủi ro rời bỏ)"
}

model_package = {
    "model": kmeans,
    "scaler": scaler,
    "pca": pca,
    "features": df_original.select_dtypes(include=[np.number]).columns.tolist(),
    "cluster_desc": cluster_desc
}

# Lưu file cùng thư mục với app.py
output_path = '../models/final_model.pkl'
joblib.dump(model_package, output_path)

print(f"✅ THÀNH CÔNG! Đã tạo file '{output_path}' dùng NumPy {np.__version__}")