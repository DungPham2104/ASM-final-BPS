import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
import io

# Bỏ qua các cảnh báo
warnings.filterwarnings('ignore')

# --- Tiêu đề ứng dụng ---
st.title("Phân tích Dữ liệu Bán hàng theo mô hình 6P")
st.markdown("Ứng dụng này phân tích dữ liệu bán hàng và dự đoán số lượng bán ra bằng mô hình Random Forest.")

# --- Bước 1: Tải và tiền xử lý dữ liệu ---
st.header("1. Tải và Tiền xử lý Dữ liệu")

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Hàm tải và tiền xử lý dữ liệu"""
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.success("Đã tải file CSV thành công.")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}. Vui lòng kiểm tra định dạng file.")
            return None
    else:
        st.info("Chưa có file nào được tải lên. Đang sử dụng dữ liệu mẫu.")
        data = {
            'Date': pd.to_datetime(['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-08', '2024-06-09', '2024-06-10', '2024-06-11']),
            'SKU': ['SKU001', 'SKU002', 'SKU001', 'SKU003', 'SKU002', 'SKU001', 'SKU004', 'SKU005', 'SKU003', 'SKU004', 'SKU005'],
            'Order_Quantity': [10, 15, 12, 20, 18, 15, 25, 30, 22, 28, 35],
            'Stock_Level': [50, 40, 38, 60, 42, 35, 55, 65, 58, 62, 70],
            'Unit_Price': [100.50, 120.00, 100.50, 100.50, 120.00, 100.50, 150.00, 95.50, 85.75, 150.00, 95.50],
            'Total_Amount': [1005.00, 1800.00, 1206.00, 2010.00, 2160.00, 1507.50, 3750.00, 2865.00, 1886.5, 4200.0, 3342.5]
        }
        df = pd.DataFrame(data)

    # Tiền xử lý dữ liệu
    required_cols = ['Date', 'SKU', 'Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
    if not all(col in df.columns for col in required_cols):
        st.error(f"File CSV thiếu các cột bắt buộc: {required_cols}. Vui lòng kiểm tra lại file.")
        return None

    numeric_cols = ['Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Date', 'SKU'], inplace=True)
    
    st.write("5 dòng dữ liệu đầu tiên:")
    st.dataframe(df.head())
    return df

uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
df = load_and_preprocess_data(uploaded_file)

if df is not None:
    time_series_df = df.groupby('Date').agg({
        'Order_Quantity': 'sum',
        'Total_Amount': 'sum'
    }).reset_index()
    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
    time_series_df.set_index('Date', inplace=True)

    # --- Bước 2: Phân tích Dữ liệu và Trực quan hóa (theo mô hình 6P) ---
    st.header("2. Phân tích Dữ liệu Bán hàng")

    # --- Phân tích P1: PRODUCT (Sản phẩm) ---
    st.subheader("P1. Phân tích Sản phẩm")
    sku_order_quantity = df.groupby('SKU')['Order_Quantity'].sum()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sku_order_quantity.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Tổng số lượng bán ra theo Sản phẩm (SKU)")
    ax1.set_xlabel("SKU")
    ax1.set_ylabel("Tổng số lượng bán ra")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # --- Phân tích P2: PRICE (Giá cả) ---
    st.subheader("P2. Phân tích Giá cả")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(df['Unit_Price'], df['Total_Amount'], alpha=0.7, color='green')
    ax2.set_title("Mối quan hệ giữa Giá đơn vị và Tổng doanh thu")
    ax2.set_xlabel("Giá đơn vị")
    ax2.set_ylabel("Tổng doanh thu")
    ax2.grid(True)
    st.pyplot(fig2)
    
    # --- Phân tích P3: PLACE/INVENTORY (Tồn kho) ---
    st.subheader("P3. Phân tích Tồn kho")
    sku_stock_level = df.groupby('SKU')['Stock_Level'].sum()
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.pie(sku_stock_level, labels=sku_stock_level.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.viridis.colors)
    ax3.set_title("Phân bổ mức tồn kho theo Sản phẩm (SKU)")
    ax3.axis('equal')
    st.pyplot(fig3)

    # --- Phân tích P4: PROMOTION (Khuyến mãi) / Xu hướng Doanh thu ---
    st.subheader("P4. Phân tích Xu hướng Doanh thu")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    time_series_df['Total_Amount'].plot(kind='line', marker='o', linestyle='-', color='red', ax=ax4)
    ax4.set_title("Xu hướng Tổng Doanh thu theo Thời gian")
    ax4.set_xlabel("Ngày")
    ax4.set_ylabel("Tổng Doanh thu")
    ax4.grid(True)
    st.pyplot(fig4)

    # --- Phân tích P5: PEOPLE / Số lượng bán ra ---
    st.subheader("P5. Phân tích Xu hướng Bán ra")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    time_series_df['Order_Quantity'].plot(kind='line', marker='o', linestyle='-', color='orange', ax=ax5)
    ax5.set_title("Xu hướng Tổng số lượng bán ra theo Thời gian")
    ax5.set_xlabel("Ngày")
    ax5.set_ylabel("Tổng số lượng bán ra")
    ax5.grid(True)
    st.pyplot(fig5)
    
    # --- Bước 3: Huấn luyện mô hình Random Forest và Dự báo ---
    st.header("3. Dự báo với Random Forest")

    if len(time_series_df) > 7:
        with st.spinner("Đang huấn luyện mô hình và dự báo..."):
            forecast_df = time_series_df.copy()
            lags = 7
            for i in range(1, lags + 1):
                forecast_df[f'lag_{i}'] = forecast_df['Order_Quantity'].shift(i)
            
            forecast_df.dropna(inplace=True)

            X = forecast_df[[f'lag_{i}' for i in range(1, lags + 1)]]
            y = forecast_df['Order_Quantity']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            forecast_steps = 7
            forecast_list = []
            last_data_point = forecast_df.iloc[-1]
            input_features = np.array([last_data_point[f'lag_{i}'] for i in range(1, lags + 1)]).reshape(1, -1)

            for _ in range(forecast_steps):
                prediction = model.predict(input_features)
                forecast_list.append(prediction[0])
                input_features = np.roll(input_features, 1)
                input_features[0, 0] = prediction[0]

            last_date = time_series_df.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=forecast_steps, freq='D')
            forecast_series = pd.Series(forecast_list, index=forecast_index)

            st.success("Dự báo hoàn thành!")
            st.write("Kết quả dự báo 7 ngày tiếp theo:")
            st.dataframe(forecast_series.reset_index().rename(columns={'index': 'Ngày', 0: 'Số lượng dự báo'}))

            # --- Trực quan hóa kết quả Dự báo ---
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            ax6.plot(time_series_df.index, time_series_df['Order_Quantity'], label='Lịch sử', color='blue', marker='o')
            ax6.plot(forecast_series.index, forecast_series.values, label='Dự báo Random Forest', color='red', linestyle='--', marker='x')
            ax6.set_title("Dự báo Số lượng Bán ra (Random Forest)")
            ax6.set_xlabel("Ngày")
            ax6.set_ylabel("Số lượng bán ra")
            ax6.legend()
            ax6.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig6)
    else:
        st.warning("Dữ liệu không đủ để tạo các đặc trưng (features) cho mô hình. Cần tối thiểu 8 ngày dữ liệu.")
