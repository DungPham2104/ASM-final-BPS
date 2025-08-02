import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ABC Sales Dashboard", layout="wide")
st.title("📊 ABC Manufacturing Data Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    df['PurchaseMonth'] = df['PurchaseDate'].dt.month
    df['PurchaseDayOfWeek'] = df['PurchaseDate'].dt.dayofweek
    df['ProductEncoded'] = LabelEncoder().fit_transform(df['Product'])
    df['RegionEncoded'] = LabelEncoder().fit_transform(df['Region'])
    scaler = StandardScaler()
    df[['SalesAmount_scaled', 'CustomerRating_scaled']] = scaler.fit_transform(df[['SalesAmount', 'CustomerRating']])

    st.header("📈 Data Visualization")

    st.markdown("### 🛍️ Sales by Product")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df, x='Product', y='SalesAmount', ax=ax1)
    st.pyplot(fig1)
    st.markdown("- Shows **sales variation** across product types.")

    st.markdown("### 🌍 Total Sales by Region")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x='Region', y='SalesAmount', estimator=sum, ci=None, ax=ax2)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)
    st.markdown("- Compares **total revenue** by region.")

    st.markdown("### ⭐ Customer Rating Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df['CustomerRating'], bins=5, kde=False, ax=ax3)
    st.pyplot(fig3)
    st.markdown("- Distribution of customer satisfaction ratings.")

    st.markdown("### ⏳ Sales Over Time")
    df_sorted = df.sort_values('PurchaseDate')
    fig4, ax4 = plt.subplots()
    sns.lineplot(data=df_sorted, x='PurchaseDate', y='SalesAmount', ax=ax4)
    st.pyplot(fig4)
    st.markdown("- Trend analysis over time.")

    st.markdown("### 📈 Average Rating per Product")
    avg_rating = df.groupby('Product')['CustomerRating'].mean().reset_index()
    fig5, ax5 = plt.subplots()
    sns.barplot(data=avg_rating, x='Product', y='CustomerRating', ax=ax5)
    st.pyplot(fig5)
    st.markdown("- Identifies well-rated vs. poorly-rated products.")

    st.subheader("🤖 Sales Prediction Model (Experimental)")
    features = ['ProductEncoded', 'RegionEncoded', 'CustomerRating', 'PurchaseMonth', 'PurchaseDayOfWeek']
    X = df[features]
    y = df['SalesAmount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("📉 Mean Squared Error (MSE):", round(mse, 2))
    st.write("📊 R² Score:", round(r2, 2))
else:
    st.warning("📂 Please upload a CSV file to begin.")

