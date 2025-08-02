import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import base64

# Tiêu đề ứng dụng
st.title("Phân tích và Dự báo Dữ liệu ABC Manufacturing")

# Bước 1: Tải và tiền xử lý dữ liệu
st.header("Tiền xử lý dữ liệu")
@st.cache_data
def load_and_preprocess_data():
    # Đọc file CSV
    df = pd.read_csv('orders_sample_with_stock.csv')
    # Hiển thị 5 dòng đầu tiên
    st.write("5 dòng đầu tiên:", df.head())
    # Kiểm tra giá trị thiếu
    st.write("Giá trị thiếu:", df.isnull().sum())
    # Điền giá trị thiếu
    df['Order_Quantity'].fillna(df['Order_Quantity'].mean(), inplace=True)
    df['Stock_Level'].fillna(df['Stock_Level'].mean(), inplace=True)
    df['Unit_Price'].fillna(df['Unit_Price'].mean(), inplace=True)
    df['Total_Amount'].fillna(df['Order_Quantity'] * df['Unit_Price'], inplace=True)
    df.dropna(subset=['Date', 'SKU'], inplace=True)
    # Chuẩn hóa dữ liệu
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df[['Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']] = df[['Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']].apply(pd.to_numeric, errors='coerce')
    df.drop_duplicates(inplace=True)
    return df

df = load_and_preprocess_data()
st.write("Dữ liệu đã xử lý:", df.head())

# Bước 2: Mô hình ARIMA
st.header("Mô hình ARIMA")
@st.cache_data
def fit_arima_model():
    series = df.groupby('Date')['Order_Quantity'].sum()
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    return forecast

forecast = fit_arima_model()
st.write("Dự báo 7 ngày tiếp theo:", forecast)

# Bước 3: Trực quan hóa 5 biểu đồ (JSON config cho Chart.js)
st.header("Trực quan hóa dữ liệu")

# Biểu đồ 1: Bar Chart
st.subheader("1. Tổng số lượng đặt hàng theo SKU")
st.json({
    "type": "bar",
    "data": {
        "labels": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "datasets": [{
            "label": "Tổng số lượng đặt hàng",
            "data": [29, 20, 23, 45, 10],
            "backgroundColor": "#FF6384",
            "borderColor": "#FF6384",
            "borderWidth": 1
        }]
    },
    "options": {
        "scales": {"y": {"beginAtZero": true, "title": {"display": true, "text": "Số lượng"}}},
        "plugins": {"title": {"display": true, "text": "Tổng số lượng đặt hàng theo SKU"}}
    }
})

# Biểu đồ 2: Line Chart
st.subheader("2. Xu hướng đặt hàng theo ngày")
st.json({
    "type": "line",
    "data": {
        "labels": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05"],
        "datasets": [{
            "label": "Số lượng đặt hàng",
            "data": [25, 20, 25, 25, 32],
            "backgroundColor": "rgba(54, 162, 235, 0.2)",
            "borderColor": "rgba(54, 162, 235, 1)",
            "borderWidth": 2,
            "fill": true
        }]
    },
    "options": {
        "scales": {"y": {"beginAtZero": true, "title": {"display": true, "text": "Số lượng"}}},
        "plugins": {"title": {"display": true, "text": "Xu hướng đặt hàng theo ngày"}}
    }
})

# Biểu đồ 3: Pie Chart
st.subheader("3. Phân bổ tồn kho theo SKU")
st.json({
    "type": "pie",
    "data": {
        "labels": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "datasets": [{
            "label": "Tồn kho",
            "data": [119, 55, 25, 95, 40],
            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
        }]
    },
    "options": {"plugins": {"title": {"display": true, "text": "Phân bổ tồn kho theo SKU"}}}
})

# Biểu đồ 4: Scatter Chart
st.subheader("4. Mối quan hệ giữa Unit_Price và Total_Amount")
st.json({
    "type": "scatter",
    "data": {
        "datasets": [{
            "label": "Unit Price vs Total Amount",
            "data": [
                {"x": 100.50, "y": 1005.00},
                {"x": 200.75, "y": 3011.25},
                {"x": 150.00, "y": 1200.00},
                {"x": 75.25, "y": 1505.00},
                {"x": 120.00, "y": 1200.00}
            ],
            "backgroundColor": "rgba(75, 192, 192, 0.6)"
        }]
    },
    "options": {
        "scales": {
            "x": {"title": {"display": true, "text": "Unit Price"}},
            "y": {"title": {"display": true, "text": "Total Amount"}}
        },
        "plugins": {"title": {"display": true, "text": "Mối quan hệ giữa Unit Price và Total Amount"}}
    }
})

# Biểu đồ 5: Radar Chart
st.subheader("5. So sánh hiệu suất theo SKU")
st.json({
    "type": "radar",
    "data": {
        "labels": ["Order_Quantity", "Stock_Level"],
        "datasets": [{
            "label": "SKU001", "data": [29, 119], "backgroundColor": "rgba(255, 99, 132, 0.2)", "borderColor": "rgba(255, 99, 132, 1)"
        }, {
            "label": "SKU002", "data": [20, 55], "backgroundColor": "rgba(54, 162, 235, 0.2)", "borderColor": "rgba(54, 162, 235, 1)"
        }, {
            "label": "SKU003", "data": [23, 25], "backgroundColor": "rgba(255, 206, 86, 0.2)", "borderColor": "rgba(255, 206, 86, 1)"
        }]
    },
    "options": {
        "scales": {"r": {"beginAtZero": true, "title": {"display": true, "text": "Giá trị"}}},
        "plugins": {"title": {"display": true, "text": "So sánh hiệu suất theo SKU"}}
    }
})

# Thêm yêu cầu thư viện
st.sidebar.text("Cài đặt: pip install pandas statsmodels streamlit")
