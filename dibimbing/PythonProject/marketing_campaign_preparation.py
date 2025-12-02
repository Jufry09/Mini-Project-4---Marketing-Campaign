import pandas as pd
import numpy as np
from datetime import datetime

# Membaca dataset
df = pd.read_csv('marketing_campaign_dataset.csv')

# Eksplorasi data awal
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
# Menangani missing values
print("\nMissing values before handling:", df.isnull().sum().sum())
# Untuk numerical columns, fill dengan median
numerical_cols = ['Conversion_Rate', 'Acquisition_Cost', 'ROI', 'Clicks', 'Impressions', 'Engagement_Score']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Untuk categorical columns, fill dengan mode
categorical_cols = ['Company', 'Campaign_Type', 'Target_Audience', 'Channel_Used', 'Location', 'Language', 'Customer_Segment']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after handling:", df.isnull().sum().sum())

# Menghapus duplikat
print("\nDuplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after:", df.duplicated().sum())

# Memperbaiki inkonsistensi format
df['Company'] = df['Company'].str.strip().str.title()
df['Campaign_Type'] = df['Campaign_Type'].str.strip().str.title()
df['Location'] = df['Location'].str.strip().str.title()
df['Customer_Segment'] = df['Customer_Segment'].str.strip().str.title()

# Menangani outliers menggunakan IQR method
def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}: {len(outliers)}")

    # Cap the outliers
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    return df

# Handle outliers untuk numerical columns
for col in numerical_cols:
    if col in df.columns:
        df = handle_outliers_iqr(df, col)

# Feature Engineering
# Konversi tanggal
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Membuat kolom baru
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['Month_Year'] = df['Date'].dt.strftime('%Y-%m')

# Metrics bisnis
df['CPC'] = np.where(df['Clicks'] > 0, df['Acquisition_Cost'] / df['Clicks'], 0)
df['CTR'] = np.where(df['Impressions'] > 0, (df['Clicks'] / df['Impressions']) * 100, 0)
df['Estimated_Revenue'] = df['Acquisition_Cost'] * df['ROI']
df['Profit'] = df['Estimated_Revenue'] - df['Acquisition_Cost']

# Kategorisasi
def categorize_roi(roi):
    if roi < 4: return 'Low'
    elif roi <= 7: return 'Medium'
    else: return 'High'

def categorize_engagement(score):
    if score <= 3: return 'Low'
    elif score <= 7: return 'Medium'
    else: return 'High'

df['ROI_Category'] = df['ROI'].apply(categorize_roi)
df['Engagement_Level'] = df['Engagement_Score'].apply(categorize_engagement)

# Export data yang sudah dibersihkan
df.to_csv('marketing_campaign_cleaned.csv', index=False)
print("\nCleaned data exported to marketing_campaign_cleaned.csv")
print("Final dataset shape:", df.shape)

# Analisis Univariate
print("=== UNIVARIATE ANALYSIS ===")
print("\nConversion Rate Distribution:")
print(df['Conversion_Rate'].describe())

print("\nROI Distribution:")
print(df['ROI'].describe())

print("\nCompany Distribution:")
print(df['Company'].value_counts())

print("\nCampaign Type Distribution:")
print(df['Campaign_Type'].value_counts())

# Analisis Bivariate
print("\n=== BIVARIATE ANALYSIS ===")
print("\nROI by Company:")
print(df.groupby('Company')['ROI'].mean().sort_values(ascending=False))

print("\nConversion Rate by Campaign Type:")
print(df.groupby('Campaign_Type')['Conversion_Rate'].mean().sort_values(ascending=False))

print("\nEngagement Score by Customer Segment:")
print(df.groupby('Customer_Segment')['Engagement_Score'].mean().sort_values(ascending=False))

# Analisis Multivariate
print("\n=== MULTIVARIATE ANALYSIS ===")
correlation_matrix = df[['ROI', 'Conversion_Rate', 'Acquisition_Cost', 'Clicks', 'Impressions', 'Engagement_Score']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Agregasi dan Summarization
print("\n=== AGGREGATION AND SUMMARIZATION ===")
monthly_performance = df.groupby('Month_Year').agg({
    'ROI': 'mean',
    'Conversion_Rate': 'mean',
    'Acquisition_Cost': 'sum',
    'Clicks': 'sum',
    'Estimated_Revenue': 'sum'
}).round(2)

print("Monthly Performance Summary:")
print(monthly_performance)

company_performance = df.groupby('Company').agg({
    'ROI': ['mean', 'max', 'min'],
    'Conversion_Rate': 'mean',
    'Acquisition_Cost': 'sum',
    'Estimated_Revenue': 'sum',
    'Campaign_ID': 'count'
}).round(2)

print("\nCompany Performance Summary:")
print(company_performance)

# Data check untuk Tableau
print("\n=== DATA CHECK FOR TABLEAU ===")
print("Data Types:")
print(df.dtypes)
print("\nMissing Values Check:")
print(df.isnull().sum())
print("\nUnique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")