import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="ABC Sales Dashboard", layout="wide")
st.title("📊 ABC Manufacturing Data Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Preprocessing
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    df['PurchaseMonth'] = df['PurchaseDate'].dt.month
    df['PurchaseDayOfWeek'] = df['PurchaseDate'].dt.dayofweek
    df['ProductEncoded'] = LabelEncoder().fit_transform(df['Product'])
    df['RegionEncoded'] = LabelEncoder().fit_transform(df['Region'])
    scaler = StandardScaler()
    df[['SalesAmount_scaled', 'CustomerRating_scaled']] = scaler.fit_transform(df[['SalesAmount', 'CustomerRating']])

    st.header("📈 Data Visualization")

    # 1. Sales by Product
    st.markdown("### 🛍️ Sales by Product")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df, x='Product', y='SalesAmount', ax=ax1)
    st.pyplot(fig1)
    st.code(
        "# Phân tích:\n"
        "# - Hiển thị sự phân bố doanh số theo từng sản phẩm.\n"
        "# - Phát hiện outliers (giá trị bất thường) và sự khác biệt giữa các dòng."
    )

    # 2. Sales by Region
    st.markdown("### 🌍 Sales by Region")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x='Region', y='SalesAmount', estimator=sum, ci=None, ax=ax2)
    st.pyplot(fig2)
    st.code(
        "# Phân tích:\n"
        "# - So sánh tổng doanh số giữa các khu vực địa lý.\n"
        "# - Xác định khu vực có tiềm năng hoặc cần cải thiện."
    )

    # 3. Customer Rating Distribution
    st.markdown("### ⭐ Customer Rating Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['CustomerRating'], bins=5, kde=False, ax=ax3)
    st.pyplot(fig3)
    st.code(
        "# Phân tích:\n"
        "# - Phân bố mức độ hài lòng khách hàng (1 đến 5).\n"
        "# - Nhiều đánh giá thấp có thể chỉ ra vấn đề về chất lượng hoặc dịch vụ."
    )

    # 4. Sales Over Time
    st.markdown("### ⏳ Sales Over Time")
    df_sorted = df.sort_values('PurchaseDate')
    fig4, ax4 = plt.subplots()
    sns.lineplot(data=df_sorted, x='PurchaseDate', y='SalesAmount', ax=ax4)
    st.pyplot(fig4)
    st.code(
        "# Phân tích:\n"
        "# - Theo dõi doanh số theo thời gian.\n"
        "# - Hữu ích để phát hiện xu hướng, mùa vụ và thời điểm giảm/tăng đột biến."
    )

    # 5. Average Rating per Product
    st.markdown("### 📈 Average Rating per Product")
    avg_rating = df.groupby('Product')['CustomerRating'].mean().reset_index()
    fig5, ax5 = plt.subplots()
    sns.barplot(data=avg_rating, x='Product', y='CustomerRating', ax=ax5)
    st.pyplot(fig5)
    st.code(
        "# Phân tích:\n"
        "# - Trung bình điểm đánh giá của từng dòng sản phẩm.\n"
        "# - Giúp xác định sản phẩm được yêu thích nhất hoặc cần cải tiến."
    )

else:
    st.warning("📂 Please upload a CSV file to begin.")



