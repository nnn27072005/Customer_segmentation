import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from scipy.stats import boxcox

# --- CẤU HÌNH ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --- LOAD MODEL & RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load gói model đã lưu từ notebook
        package = joblib.load('../models/final_model.pkl')
        
        # Load thêm Scaler (Vì trong cluster_lib scaler không nằm trong analyzer, 
        # ta cần file này. Nếu bạn chưa có file này, xem phần LƯU Ý bên dưới)
        try:
            scaler = joblib.load('../data/processed/scaler.pkl') # Hoặc đường dẫn nơi bạn lưu scaler
        except:
            # Nếu không tìm thấy scaler cũ, ta tạo scaler mới (chỉ để demo không bị crash, 
            # nhưng tốt nhất là bạn nên copy file scaler.pkl vào cùng folder)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
        return package, scaler
    except FileNotFoundError:
        return None, None

model_package, scaler = load_resources()

if model_package is None:
    st.error("⚠️ Không tìm thấy file 'final_model.pkl'. Hãy chạy đoạn code lưu model ở notebook trước!")
    st.stop()

# Trích xuất thành phần
kmeans = model_package['model']
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
    
    # Tạo input cho 16 features. 
    # Để giao diện đẹp, ta nhóm các feature quan trọng lên đầu
    
    # Nhóm 1: Feature gốc quan trọng
    st.markdown("**Chỉ số cơ bản:**")
    input_data['Sum_Quantity'] = st.number_input("Tổng số lượng hàng (Sum_Quantity)", value=100.0)
    input_data['Sum_TotalPrice'] = st.number_input("Tổng chi tiêu (Sum_TotalPrice)", value=500.0)
    input_data['Count_Invoice'] = st.number_input("Số lần mua (Count_Invoice)", value=5.0)
    input_data['Count_Stock'] = st.number_input("Số loại hàng (Count_Stock)", value=10.0)
    input_data['Mean_UnitPrice'] = st.number_input("Đơn giá trung bình (Mean_UnitPrice)", value=5.0)
    
    # Nhóm 2: Các feature tính toán (cho vào expander để gọn)
    with st.expander("Các chỉ số nâng cao (Mở rộng)"):
        # Lấy các feature còn lại trong feature_names mà chưa có input
        remaining_features = [f for f in feature_names if f not in input_data]
        for f in remaining_features:
            input_data[f] = st.number_input(f"{f}", value=0.0)

    btn_predict = st.button("Phân tích ngay", type="primary")

with col2:
    if btn_predict:
        # --- BƯỚC 1: TẠO DATAFRAME ---
        # Đảm bảo thứ tự cột đúng y hệt lúc train
        df_input = pd.DataFrame([input_data])
        df_input = df_input[feature_names] # Reorder columns
        
        # --- BƯỚC 2: XỬ LÝ DỮ LIỆU (PREPROCESSING) ---
        # Logic này mô phỏng lại class FeatureEngineer trong cluster_lib.py
        
        # 2.1 Box-Cox Transformation
        # Lưu ý: Box-cox cần dữ liệu dương. Ta dùng np.log1p (log(x+1)) 
        # như một sự thay thế an toàn và gần tương đương cho Web App để tránh lỗi số âm.
        # (Trong thực tế production, bạn cần lưu lambda của boxcox từng cột để transform chính xác 100%)
        df_transformed = np.log1p(np.abs(df_input))
        
        # 2.2 Scaling
        # Nếu không có scaler xịn, bước này chỉ mang tính tượng trưng
        if scaler:
            try:
                # Nếu scaler chưa fit (trường hợp tạo mới), ta fit tạm (không khuyến khích)
                # Nếu scaler load từ file, nó sẽ transform đúng
                if hasattr(scaler, 'mean_'): 
                    X_scaled = scaler.transform(df_transformed)
                else:
                    X_scaled = scaler.fit_transform(df_transformed)
            except:
                 X_scaled = df_transformed.values # Fallback
        else:
            X_scaled = df_transformed.values

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
        
        # Chọn 5 chỉ số quan trọng nhất để vẽ
        radar_cols = ['Sum_Quantity', 'Sum_TotalPrice', 'Count_Invoice', 'Count_Stock', 'Mean_UnitPrice']
        radar_vals = [input_data[c] for c in radar_cols]
        
        # Chuẩn hóa min-max (giả lập) để vẽ lên biểu đồ cho đẹp
        # Vì giá trị tiền rất to so với số lần mua, ta dùng log để vẽ
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