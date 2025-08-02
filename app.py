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
        st.json({
            "type": "bar",
            "data": {
                "labels": list(bar_data.keys()),
                "datasets": [{
                    "label": "Tổng số lượng đặt hàng",
                    "data": list(bar_data.values()),
                    "backgroundColor": "#FF6384",
                    "borderColor": "#FF6384",
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Số lượng"}}},
                "plugins": {"title": {"display": True, "text": "Tổng số lượng đặt hàng theo SKU"}}
            }
        })

    # Biểu đồ 2: Line Chart
    st.subheader("2. Xu hướng đặt hàng theo ngày")
    line_data = {d.strftime('%Y-%m-%d'): v for d, v in date_totals.items()}
    line_labels = sorted(line_data.keys())[:5]  # Lấy 5 ngày đầu
    line_values = [line_data.get(d, 0) for d in line_labels]
    if not line_labels:
        st.warning("Không có dữ liệu để vẽ biểu đồ đường.")
    else:
        st.json({
            "type": "line",
            "data": {
                "labels": line_labels,
                "datasets": [{
                    "label": "Số lượng đặt hàng",
                    "data": line_values,
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                    "fill": True
                }]
            },
            "options": {
                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Số lượng"}}},
                "plugins": {"title": {"display": True, "text": "Xu hướng đặt hàng theo ngày"}}
            }
        })

    # Biểu đồ 3: Pie Chart
    st.subheader("3. Phân bổ tồn kho theo SKU")
    pie_data = {k: stock_levels.get(k, 0) for k in ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'SKU005']}
    if not pie_data or all(v == 0 for v in pie_data.values()):
        st.warning("Không có dữ liệu để vẽ biểu đồ tròn.")
    else:
        st.json({
            "type": "pie",
            "data": {
                "labels": list(pie_data.keys()),
                "datasets": [{
                    "label": "Tồn kho",
                    "data": list(pie_data.values()),
                    "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                }]
            },
            "options": {"plugins": {"title": {"display": True, "text": "Phân bổ tồn kho theo SKU"}}}
        })

    # Biểu đồ 4: Scatter Chart
    st.subheader("4. Mối quan hệ giữa Unit_Price và Total_Amount")
    scatter_data = [{"x": row[0], "y": row[1]} for row in price_amount_pairs][:5]  # Lấy 5 cặp đầu
    if not scatter_data:
        st.warning("Không có dữ liệu để vẽ biểu đồ phân tán.")
    else:
        st.json({
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": "Unit Price vs Total Amount",
                    "data": scatter_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.6)"
                }]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Unit Price"}},
                    "y": {"title": {"display": True, "text": "Total Amount"}}
                },
                "plugins": {"title": {"display": True, "text": "Mối quan hệ giữa Unit Price và Total Amount"}}
            }
        })

    # Biểu đồ 5: Radar Chart
    st.subheader("5. So sánh hiệu suất theo SKU")
    radar_data = {
        "SKU001": [sku_totals.get('SKU001', 0), stock_levels.get('SKU001', 0)],
        "SKU002": [sku_totals.get('SKU002', 0), stock_levels.get('SKU002', 0)],
        "SKU003": [sku_totals.get('SKU003', 0), stock_levels.get('SKU003', 0)]
    }
    if not any(any(v) for v in radar_data.values()):
        st.warning("Không có dữ liệu để vẽ biểu đồ radar.")
    else:
        st.json({
            "type": "radar",
            "data": {
                "labels": ["Order_Quantity", "Stock_Level"],
                "datasets": [
                    {"label": k, "data": v, "backgroundColor": "rgba(255, 99, 132, 0.2)", "borderColor": "rgba(255, 99, 132, 1)"}
                    for k, v in radar_data.items() if any(v)
                ]
            },
            "options": {
                "scales": {"r": {"beginAtZero": True, "title": {"display": True, "text": "Giá trị"}}},
                "plugins": {"title": {"display": True, "text": "So sánh hiệu suất theo SKU"}}
            }
        })

# Yêu cầu thư viện
st.sidebar.text("Cài đặt: pip install pandas statsmodels streamlit")
