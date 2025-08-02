import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import HessianInversionWarning
import warnings

# Bỏ qua cảnh báo về Hessian
warnings.filterwarnings('ignore', category=HessianInversionWarning)

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
            # Tạo file CSV mẫu nếu không có file mặc định và không có file tải lên
            st.warning("Không tìm thấy file 'orders_sample_with_stock.csv'. Đang tạo dữ liệu mẫu...")
            data = {
                'Date': ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-08'],
                'SKU': ['SKU001', 'SKU002', 'SKU001', 'SKU003', 'SKU002', 'SKU001', 'SKU004', 'SKU005'],
                'Order_Quantity': [10, 15, 12, 20, 18, 15, 25, 30],
                'Stock_Level': [50, 40, 38, 60, 42, 35, 55, 65],
                'Unit_Price': [100.50, 120.00, 100.50, 85.75, 120.00, 100.50, 150.00, 95.50],
                'Total_Amount': [1005.00, 1800.00, 1206.00, 1715.00, 2160.00, 1507.50, 3750.00, 2865.00]
            }
            df = pd.DataFrame(data)
            st.info("Đã tạo dữ liệu mẫu để tiếp tục.")
            
        # Kiểm tra định dạng cơ bản
        required_columns = ['Date', 'SKU', 'Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
        current_columns = df.columns.tolist()
        if not all(col in current_columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in current_columns]
            st.error(f"File CSV thiếu cột yêu cầu. Cần: {required_columns}. Thiếu: {missing_cols}. Cột hiện tại: {current_columns}")
            st.warning("File hiện tại có thể là metadata. Vui lòng tải lên file CSV chứa dữ liệu thực tế.")
            return None

        # Hiển thị 5 dòng đầu tiên
        st.write("5 dòng đầu tiên:", df.head())

        # Kiểm tra giá trị thiếu
        st.write("Giá trị thiếu:", df.isnull().sum())

        # Điền giá trị thiếu và chuẩn hóa
        numeric_cols = ['Order_Quantity', 'Stock_Level', 'Unit_Price', 'Total_Amount']
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        df.dropna(subset=['Date', 'SKU'], inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df.drop_duplicates(inplace=True)

        if df.empty:
            st.error("Dữ liệu sau khi xử lý rỗng. Vui lòng kiểm tra file CSV.")
            return None

        st.success("Dữ liệu đã được tiền xử lý thành công.")
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
st.header("Mô hình ARIMA và Dự báo")
@st.cache_data
def fit_arima_model(df):
    try:
        # Chuẩn bị chuỗi time-series
        series = df.groupby('Date')['Order_Quantity'].sum()
        if len(series) < 3:
            st.warning("Dữ liệu không đủ để huấn luyện mô hình ARIMA. Cần ít nhất 3 điểm dữ liệu theo ngày.")
            return None
        
        # Huấn luyện mô hình ARIMA
        # order=(p,d,q) - p:AR, d:Integrated, q:MA
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Dự báo 7 ngày tiếp theo
        forecast = model_fit.forecast(steps=7)
        return series, forecast
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện ARIMA: {e}")
        return None, None

if df is not None:
    series, forecast = fit_arima_model(df)
    if forecast is not None:
        st.subheader("Dự báo 7 ngày tiếp theo")
        st.dataframe(forecast.reset_index().rename(columns={'index': 'Ngày', 'predicted_mean': 'Số lượng dự báo'}))

# Bước 3: Trực quan hóa 5 biểu đồ
st.header("Trực quan hóa dữ liệu")
if df is not None and not df.empty:
    
    # Biểu đồ 1: Tổng số lượng đặt hàng theo SKU (Bar Chart)
    st.subheader("1. Tổng số lượng đặt hàng theo SKU")
    sku_order_quantity = df.groupby('SKU')['Order_Quantity'].sum()
    if not sku_order_quantity.empty:
        fig1, ax1 = plt.subplots()
        ax1.bar(sku_order_quantity.index, sku_order_quantity.values, color='#1f77b4')
        ax1.set_title("Tổng số lượng đặt hàng theo SKU")
        ax1.set_xlabel("SKU")
        ax1.set_ylabel("Tổng số lượng")
        st.pyplot(fig1)
    else:
        st.warning("Không có dữ liệu để vẽ biểu đồ cột.")

    # Biểu đồ 2: Tổng số lượng đặt hàng theo thời gian (Line Chart)
    st.subheader("2. Tổng số lượng đặt hàng theo thời gian")
    if series is not None and not series.empty:
        fig2, ax2 = plt.subplots()
        ax2.plot(series.index, series.values, marker='o', linestyle='-', color='#ff7f0e')
        ax2.set_title("Tổng số lượng đặt hàng theo thời gian")
        ax2.set_xlabel("Ngày")
        ax2.set_ylabel("Tổng số lượng")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.warning("Không có đủ dữ liệu chuỗi thời gian để vẽ biểu đồ đường.")

    # Biểu đồ 3: Mức tồn kho theo SKU (Pie Chart)
    st.subheader("3. Mức tồn kho theo SKU")
    sku_stock_level = df.groupby('SKU')['Stock_Level'].sum()
    if not sku_stock_level.empty:
        fig3, ax3 = plt.subplots()
        ax3.pie(sku_stock_level, labels=sku_stock_level.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax3.axis('equal') # Đảm bảo hình tròn
        ax3.set_title("Mức tồn kho theo SKU")
        st.pyplot(fig3)
    else:
        st.warning("Không có dữ liệu để vẽ biểu đồ tròn.")

    # Biểu đồ 4: Mối quan hệ giữa giá và tổng tiền (Scatter Plot)
    st.subheader("4. Mối quan hệ giữa giá và tổng tiền")
    if not df.empty:
        fig4, ax4 = plt.subplots()
        ax4.scatter(df['Unit_Price'], df['Total_Amount'], alpha=0.5, color='#2ca02c')
        ax4.set_title("Mối quan hệ giữa Giá Đơn vị và Tổng Tiền")
        ax4.set_xlabel("Giá Đơn vị")
        ax4.set_ylabel("Tổng Tiền")
        st.pyplot(fig4)
    else:
        st.warning("Không có dữ liệu để vẽ biểu đồ phân tán.")

    # Biểu đồ 5: Dự báo số lượng đặt hàng (Line Chart với Dự báo)
    st.subheader("5. Dự báo số lượng đặt hàng")
    if series is not None and forecast is not None:
        fig5, ax5 = plt.subplots()
        ax5.plot(series.index, series.values, label='Lịch sử', color='blue', marker='o')
        forecast_index = pd.date_range(start=series.index[-1] + pd.DateOffset(days=1), periods=7, freq='D')
        ax5.plot(forecast_index, forecast, label='Dự báo', color='red', linestyle='--', marker='x')
        ax5.set_title("Dự báo Số lượng Đặt hàng với ARIMA")
        ax5.set_xlabel("Ngày")
        ax5.set_ylabel("Số lượng")
        ax5.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig5)
    else:
        st.warning("Không có dữ liệu lịch sử hoặc dự báo để vẽ biểu đồ.")
