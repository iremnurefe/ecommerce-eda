import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="E-Commerce EDA Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('online_retail_II.csv', encoding='utf-8')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Customer ID'] = df['Customer ID'].astype(str).str.replace('.0', '', regex=False)
    df['Customer ID'] = df['Customer ID'].replace('nan', np.nan)
    df = df[~df['Invoice'].str.startswith('C')]
    df = df[(df['Price'] > 0) & (df['Quantity'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    exclude = ['Manual', 'DOTCOM POSTAGE', 'POSTAGE', 'AMAZONFEE']
    df = df[~df['Description'].isin(exclude)]
    return df

df = load_data()

@st.cache_data
def compute_rfm(df):
    df_rfm = df.dropna(subset=['Customer ID']).copy()
    reference_date = df_rfm['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_rfm.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5])

    def segment_customer(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:
            return 'VIP'
        elif r >= 4 and f >= 3:
            return 'Loyal'
        elif r >= 4 and f <= 2:
            return 'New Customer'
        elif r == 3:
            return 'Potential'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        else:
            return 'Churned'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

rfm = compute_rfm(df)

# Header
st.title("🛒 E-Commerce Sales Dashboard")
st.markdown("**Online Retail II** dataset analysis — 2009-2011")
st.divider()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
col2.metric("📦 Total Orders", f"{df['Invoice'].nunique():,}")
col3.metric("👥 Unique Customers", f"{df['Customer ID'].nunique():,}")
col4.metric("🧾 Avg. Order Value", f"£{df.groupby('Invoice')['TotalPrice'].sum().mean():,.0f}")

st.divider()

# Monthly Revenue Trend
st.subheader("📈 Monthly Revenue Trend")
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
monthly = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
fig1 = px.line(monthly, x='YearMonth', y='TotalPrice',
               labels={'YearMonth': 'Month', 'TotalPrice': 'Revenue (£)'}, markers=True)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Country and Product side by side
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌍 Top 10 Countries (Excl. UK)")
    country_rev = df[df['Country'] != 'United Kingdom'].groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig2 = px.bar(x=country_rev.values, y=country_rev.index, orientation='h',
                  labels={'x': 'Revenue (£)', 'y': 'Country'},
                  color=country_rev.values, color_continuous_scale='Blues')
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("🏆 Top Products by Revenue")
    top_prod = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig3 = px.bar(x=top_prod.values, y=top_prod.index, orientation='h',
                  labels={'x': 'Revenue (£)', 'y': 'Product'},
                  color=top_prod.values, color_continuous_scale='Greens')
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# RFM Segmentation
st.subheader("👤 Customer Segments (RFM Analysis)")
col_pie, col_table = st.columns(2)

with col_pie:
    seg_counts = rfm['Segment'].value_counts()
    fig4 = px.pie(values=seg_counts.values, names=seg_counts.index,
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig4, use_container_width=True)

with col_table:
    seg_summary = rfm.groupby('Segment').agg(
        Customer_Count=('Customer ID', 'count'),
        Avg_Spend=('Monetary', 'mean'),
        Avg_Orders=('Frequency', 'mean')
    ).round(1).sort_values('Customer_Count', ascending=False)
    st.dataframe(seg_summary, use_container_width=True)
