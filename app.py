import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
from datetime import datetime

# Tiêu đề ứng dụng
st.title("Phân tích và Dự báo Dữ liệu ABC Manufacturing")

# Bước 1: Tải và tiền xử lý dữ liệu
st.header("Tiền xử lý dữ liệu")
@st.cache_data
def load_and_preprocess_data(uploaded_file=None):
    df = None
    try:
        # Kiểm tra và đọc file CSV với xử lý lỗi dòng
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.success("Đã tải file CSV thành công.")
        elif os.path.exists('orders_sample_with_stock.csv'):
            df = pd.read_csv('orders_sample_with_stock.csv', on_bad_lines='skip')
            st.success("Đã sử dụng file mặc định 'orders_sample_with_stock.csv'.")
        else:
            st.error("File 'orders_sample_with_stock.csv' không tồn tại. Vui lòng tải lên file CSV.")
            return None

        # Kiểm tra định dạng cơ bản
        required_columns = ['Date', 'SKU', 'Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
        current_columns = df.columns.tolist()
        if not all(col in current_columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in current_columns]
            st.error(f"File CSV thiếu cột yêu cầu. Cần: {required_columns}. Thiếu: {missing_cols}. Cột hiện tại: {current_columns}")
            st.warning("File hiện tại có thể là metadata (ví dụ: 'Cột', 'Kiểu dữ liệu', 'Mô tả'). Vui lòng tải lên file CSV chứa dữ liệu thực tế với 6 cột: Date, SKU, Order_Quantity, Stock_Level, Unit_Price, Total_Amount.")
            st.info("Ví dụ định dạng:\nDate,SKU,Order_Quantity,Stock_Level,Unit_Price,Total_Amount\n2024-06-01,SKU001,10,50,100.50,1005.00")
            return None

        # Hiển thị 5 dòng đầu tiên
        st.write("5 dòng đầu tiên:", df.head())

        # Kiểm tra giá trị thiếu
        st.write("Giá trị thiếu:", df.isnull().sum())

        # Điền giá trị thiếu
        numeric_cols = ['Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        df['Total_Amount'].fillna(df['Order_Quantity'] * df['Unit_Price'], inplace=True)
        df.dropna(subset=['Date', 'SKU'], inplace=True)

        # Chuẩn hóa dữ liệu
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        current_date = datetime(2025, 8, 3)  # Thời gian hiện tại
        df = df[df['Date'] <= current_date]  # Loại bỏ ngày tương lai
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df.drop_duplicates(inplace=True)

        # Kiểm tra dữ liệu rỗng
        if df.empty:
            st.error("Dữ liệu sau khi xử lý rỗng. Vui lòng kiểm tra file CSV.")
            return None

        # Lưu dữ liệu đã xử lý (tuỳ chọn)
        df.to_csv('orders_processed.csv', index=False)
        st.success("Dữ liệu đã được tiền xử lý và lưu vào 'orders_processed.csv'")
        return df
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        return None

# Tải file hoặc sử dụng file mặc định
uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
df = load_and_preprocess_data(uploaded_file)
if df is None or df.empty:
    st.stop()

# Bước 2: Mô hình ARIMA
st.header("Mô hình ARIMA")
@st.cache_data
def fit_arima_model(df):
    try:
        # Chuẩn bị chuỗi time-series
        series = df.groupby('Date')['Order_Quantity'].sum()
        if len(series) < 3:
            st.warning("Dữ liệu không đủ để huấn luyện mô hình ARIMA. Cần ít nhất 3 điểm dữ liệu theo ngày.")
            return None
        # Huấn luyện mô hình ARIMA
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        # Dự báo 7 ngày tiếp theo
        forecast = model_fit.forecast(steps=7)
        st.write("Dự báo 7 ngày tiếp theo:", forecast)
        return forecast
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện ARIMA: {e}")
        return None

if df is not None:
    forecast = fit_arima_model(df)

# Bước 3: Trực quan hóa 5 biểu đồ
st.header("Trực quan hóa dữ liệu")
if df is not None and not df.empty:
    # Tổng hợp dữ liệu cho biểu đồ
    sku_totals = df.groupby('SKU')['Order_Quantity'].sum().to_dict()
    date_totals = df.groupby('Date')['Order_Quantity'].sum().to_dict()
    stock_levels = df.groupby('SKU')['Stock_Level'].sum().to_dict()
    price_amount_pairs = df[['Unit_Price', 'Total_Amount']].dropna().to_records(index=False)

    # Biểu đồ 1: Bar Chart
    st.subheader("1. Tổng số lượng đặt hàng theo SKU")
    bar_data = {k: sku_totals.get(k, 0) for k in ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']}
    if not bar_data or all(v == 0 for v in bar_data.values()):
        st.warning("Không có dữ liệu để vẽ biểu đồ cột.")
    else:
        ```chartjs
        {
            "type": "bar",
            "data": {
                "labels": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
                "datasets": [{
                    "label": "Tổng số lượng đặt hàng",
                    "data": [bar_data["SKU001"], bar_data["SKU002"], bar_data["SKU003"], bar_data["SKU004"], bar_data["SKU005"]],
                    "backgroundColor": "#FF6384",
                    "borderColor": "#FF6384",
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {"y": {"beginAtZero": true, "title": {"display": true, "text": "Số lượng"}}},
                "plugins": {"title": {"display": true, "text": "Tổng số lượng đặt hàng theo SKU"}}
            }
        }
