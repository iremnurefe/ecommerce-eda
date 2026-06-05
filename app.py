import streamlit as st
import pandas as pd
import numpy as np
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

# RFM hesapla
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
            return 'Sadık Müşteri'
        elif r >= 4 and f <= 2:
            return 'Yeni Müşteri'
        elif r == 3:
            return 'Potansiyel'
        elif r <= 2 and f >= 3:
            return 'Kaybedilmek Üzere'
        else:
            return 'Kayıp'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

rfm = compute_rfm(df)

# Başlık
st.title("🛒 E-Commerce Sales Dashboard")
st.markdown("**Online Retail II** dataset analizi — 2009-2011")
st.divider()

# KPI Kartları
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Toplam Gelir", f"£{df['TotalPrice'].sum():,.0f}")
col2.metric("📦 Toplam Sipariş", f"{df['Invoice'].nunique():,}")
col3.metric("👥 Müşteri Sayısı", f"{df['Customer ID'].nunique():,}")
col4.metric("🧾 Ort. Sipariş Değeri", f"£{df.groupby('Invoice')['TotalPrice'].sum().mean():,.0f}")

st.divider()

# Aylık Gelir Trendi
st.subheader("📈 Aylık Gelir Trendi")
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
monthly = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
fig1 = px.line(monthly, x='YearMonth', y='TotalPrice',
               labels={'YearMonth': 'Ay', 'TotalPrice': 'Gelir (£)'}, markers=True)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Ülke ve Ürün yan yana
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌍 Top 10 Ülke (UK Hariç)")
    country_rev = df[df['Country'] != 'United Kingdom'].groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig2 = px.bar(x=country_rev.values, y=country_rev.index, orientation='h',
                  labels={'x': 'Gelir (£)', 'y': 'Ülke'},
                  color=country_rev.values, color_continuous_scale='Blues')
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("🏆 En Çok Gelir Getiren Ürünler")
    top_prod = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig3 = px.bar(x=top_prod.values, y=top_prod.index, orientation='h',
                  labels={'x': 'Gelir (£)', 'y': 'Ürün'},
                  color=top_prod.values, color_continuous_scale='Greens')
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# RFM
st.subheader("👤 Müşteri Segmentleri (RFM)")
col_pie, col_table = st.columns(2)

with col_pie:
    seg_counts = rfm['Segment'].value_counts()
    fig4 = px.pie(values=seg_counts.values, names=seg_counts.index,
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig4.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig4, use_container_width=True)

with col_table:
    seg_summary = rfm.groupby('Segment').agg(
        Müşteri_Sayısı=('Customer ID', 'count'),
        Ort_Harcama=('Monetary', 'mean'),
        Ort_Sipariş=('Frequency', 'mean')
    ).round(1).sort_values('Müşteri_Sayısı', ascending=False)
    st.dataframe(seg_summary, use_container_width=True)