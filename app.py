import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 ABC Manufacturing Data Analysis App")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Tiền xử lý
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    df['PurchaseMonth'] = df['PurchaseDate'].dt.month
    df['PurchaseDayOfWeek'] = df['PurchaseDate'].dt.dayofweek

    le_product = LabelEncoder()
    df['ProductEncoded'] = le_product.fit_transform(df['Product'])

    le_region = LabelEncoder()
    df['RegionEncoded'] = le_region.fit_transform(df['Region'])

    scaler = StandardScaler()
    df[['SalesAmount_scaled', 'CustomerRating_scaled']] = scaler.fit_transform(df[['SalesAmount', 'CustomerRating']])

    st.subheader("📈 Visualization")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    sns.boxplot(data=df, x='Product', y='SalesAmount', ax=axes[0, 0])
    axes[0, 0].set_title('Sales by Product')

    sns.barplot(data=df, x='Region', y='SalesAmount', estimator=sum, ci=None, ax=axes[0, 1])
    axes[0, 1].set_title('Sales by Region')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.histplot(df['CustomerRating'], bins=5, kde=False, ax=axes[1, 0])
    axes[1, 0].set_title('Rating Distribution')

    df_sorted = df.sort_values('PurchaseDate')
    sns.lineplot(data=df_sorted, x='PurchaseDate', y='SalesAmount', ax=axes[1, 1])
    axes[1, 1].set_title('Sales Over Time')

    avg_rating = df.groupby('Product')['CustomerRating'].mean().reset_index()
    sns.barplot(data=avg_rating, x='Product', y='CustomerRating', ax=axes[2, 0])
    axes[2, 0].set_title('Average Rating per Product')

    axes[2, 1].axis('off')
    st.pyplot(fig)

    st.subheader("🤖 Sales Prediction")

    # Model
    features = ['ProductEncoded', 'RegionEncoded', 'CustomerRating', 'PurchaseMonth', 'PurchaseDayOfWeek']
    X = df[features]
    y = df['SalesAmount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"📊 **R² Score:** {r2:.2f}")
