import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- CẤU HÌNH ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --- LOAD MODEL & RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load gói model (Đã chứa sẵn Model + Scaler + PCA phiên bản mới nhất)
        package = joblib.load('../models/final_model.pkl')
        return package
    except FileNotFoundError:
        return None

# Load data
model_package = load_resources()

if model_package is None:
    st.error("⚠️ Không tìm thấy file '../models/final_model.pkl'.")
    st.stop()

# --- TRÍCH XUẤT THÀNH PHẦN (Lấy trực tiếp từ gói, không load file lẻ) ---
kmeans = model_package['model']
# QUAN TRỌNG: Lấy scaler từ trong gói này để đảm bảo đồng bộ version
scaler = model_package['scaler'] 
pca = model_package['pca']
feature_names = model_package['features']
cluster_desc = model_package['cluster_desc']

# --- GIAO DIỆN ---
st.title("📊 Dự đoán Phân khúc Khách hàng")
st.markdown("Nhập các chỉ số hành vi khách hàng để phân loại.")

# Tạo 2 cột giao diện
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Nhập liệu")
    input_data = {}
    
    st.markdown("**Chỉ số cơ bản:**")
    # Giá trị mặc định demo
    input_data['Sum_Quantity'] = st.number_input("Tổng số lượng hàng (Sum_Quantity)", value=100.0)
    input_data['Sum_TotalPrice'] = st.number_input("Tổng chi tiêu (Sum_TotalPrice)", value=500.0)
    input_data['Count_Invoice'] = st.number_input("Số lần mua (Count_Invoice)", value=5.0)
    input_data['Count_Stock'] = st.number_input("Số loại hàng (Count_Stock)", value=10.0)
    input_data['Mean_UnitPrice'] = st.number_input("Đơn giá trung bình (Mean_UnitPrice)", value=5.0)
    
    with st.expander("Các chỉ số nâng cao (Mở rộng)"):
        remaining_features = [f for f in feature_names if f not in input_data]
        for f in remaining_features:
            input_data[f] = st.number_input(f"{f}", value=0.0)

    btn_predict = st.button("Phân tích ngay", type="primary")

with col2:
    if btn_predict:
        # --- BƯỚC 1: TẠO DATAFRAME ---
        df_input = pd.DataFrame([input_data])
        # Đảm bảo thứ tự cột khớp hoàn toàn với lúc train
        df_input = df_input[feature_names] 
        
        # --- BƯỚC 2: XỬ LÝ DỮ LIỆU ---
        # 2.1 Log Transform (Thay thế an toàn cho Box-Cox)
        # Dùng np.abs để tránh lỗi log số âm nếu người dùng nhập sai
        df_transformed = np.log1p(np.abs(df_input))
        
        # 2.2 Scaling (Dùng scaler xịn lấy từ model_package)
        X_scaled = scaler.transform(df_transformed)

        # 2.3 PCA Transform
        if pca:
            X_pca = pca.transform(X_scaled)
        else:
            X_pca = X_scaled

        # --- BƯỚC 3: DỰ ĐOÁN ---
        cluster_id = kmeans.predict(X_pca)[0]
        
        # --- BƯỚC 4: HIỂN THỊ KẾT QUẢ ---
        st.success(f"Kết quả phân loại: **{cluster_desc.get(cluster_id, f'Cluster {cluster_id}')}**")
        
        # --- BƯỚC 5: BIỂU ĐỒ RADAR ---
        st.subheader("Hồ sơ khách hàng")
        
        radar_cols = ['Sum_Quantity', 'Sum_TotalPrice', 'Count_Invoice', 'Count_Stock', 'Mean_UnitPrice']
        radar_vals = [input_data[c] for c in radar_cols]
        
        # Log scale để vẽ biểu đồ đẹp hơn
        radar_vals_log = np.log1p(radar_vals)
        
        df_radar = pd.DataFrame(dict(
            r=radar_vals_log,
            theta=radar_cols
        ))
        
        fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig)

    else:
        st.info("👈 Nhập thông tin bên trái để xem khách hàng thuộc nhóm nào.")