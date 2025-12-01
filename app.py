# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Mobile Device Usage Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------
# Introduction
# -----------------------
st.markdown("""
## 📘 Introduction

### **1. Dataset Overview**
**Dataset Name:** *Mobile Device Usage and User Behavior*  
**Source:** Kaggle (Valakhorasani)

This dataset provides a comprehensive look at how individuals interact with their mobile devices.  
It includes detailed user behavior metrics such as:

- 📱 **App Usage Time** (minutes per day)  
- 🔋 **Battery Drain** (mAh per day)  
- 🌐 **Data Usage** (MB per day)  
- ⏱️ **Screen-On Time** (hours per day)  
- 📲 **Number of Apps Installed**  
- 👤 **Demographic Information** such as **age** and **gender**

The goal of this dashboard is to explore user behavior patterns, analyze correlations, detect outliers,  
and visualize how demographics affect mobile usage.
""")

# -----------------------
# Load Data
# -----------------------
st.title("📱 Mobile Device Usage & Behavior Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    return df

m_d = load_data()

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender:",
    options=m_d['gender'].unique(),
    default=m_d['gender'].unique()
)

os_filter = st.sidebar.multiselect(
    "Select Operating System:",
    options=m_d['os'].unique(),
    default=m_d['os'].unique()
)

age_range = st.sidebar.slider(
    "Select Age Range:",
    int(m_d['age'].min()), int(m_d['age'].max()),
    (int(m_d['age'].min()), int(m_d['age'].max()))
)

# Apply filters
m_d_filtered = m_d[
    (m_d['gender'].isin(gender_filter)) &
    (m_d['os'].isin(os_filter)) &
    (m_d['age'].between(age_range[0], age_range[1]))
]

# -----------------------
# Data Overview
# -----------------------
st.header("📋 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Users", len(m_d_filtered))
with col2:
    st.metric("Average Age", f"{m_d_filtered['age'].mean():.1f}")
with col3:
    st.metric("Average Apps Installed", f"{m_d_filtered['apps_installed'].mean():.1f}")
with col4:
    st.metric("Average Screen Time", f"{m_d_filtered['screen_on_hours'].mean():.1f} hours")

# Show raw data option
if st.checkbox("Show Raw Data Sample"):
    st.dataframe(m_d_filtered.head(100))

# -----------------------
# Average Usage Metrics
# -----------------------
st.subheader("📊 Average Usage Metrics")
avg_metrics = m_d_filtered[['app_usage_min', 'data_usage_mb', 'battery_drain',
                            'screen_on_hours', 'apps_installed']].mean().round(2)
st.dataframe(avg_metrics)

st.markdown("#### Distribution of Metrics")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics_to_plot = ['app_usage_min', 'data_usage_mb', 'battery_drain', 'screen_on_hours', 'apps_installed']
titles = ['App Usage (min/day)', 'Data Usage (MB/day)', 'Battery Drain (mAh/day)',
          'Screen Time (hours/day)', 'Apps Installed']

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[i//3, i%3]
    sns.histplot(m_d_filtered[metric], kde=True, ax=ax, bins=20)
    ax.axvline(m_d_filtered[metric].mean(), color='red', linestyle='--',
               label=f'Mean: {m_d_filtered[metric].mean():.1f}')
    ax.set_title(f'Distribution of {title}')
    ax.legend()

plt.tight_layout()
st.pyplot(fig)

# -----------------------
# Outlier Analysis
# -----------------------
st.subheader("📊 Outlier Analysis")

def detect_outliers(column):
    Q1 = m_d_filtered[column].quantile(0.25)
    Q3 = m_d_filtered[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = m_d_filtered[(m_d_filtered[column] < lower) | (m_d_filtered[column] > upper)]
    return len(outliers), lower, upper

outlier_cols = ['app_usage_min', 'data_usage_mb', 'battery_drain', 'screen_on_hours']
outlier_data = []
for col in outlier_cols:
    count, lower, upper = detect_outliers(col)
    outlier_data.append({
        'Metric': col,
        'Outliers Count': count,
        'Normal Range Lower': f"{lower:.1f}",
        'Normal Range Upper': f"{upper:.1f}"
    })

outlier_df = pd.DataFrame(outlier_data)
st.dataframe(outlier_df)

# -----------------------
# Scatter Plots
# -----------------------
st.subheader("📈 App Usage vs Screen On Time")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=m_d_filtered, x='app_usage_min', y='screen_on_hours',
                hue='gender', alpha=0.6, ax=ax2)
ax2.set_xlabel('App Usage Time (min/day)')
ax2.set_ylabel('Screen On Time (hours/day)')
st.pyplot(fig2)

st.subheader("📈 Number of Apps Installed vs Screen Time")
apps_usage_corr = m_d_filtered['apps_installed'].corr(m_d_filtered['screen_on_hours'])
st.write(f"Correlation coefficient: **{apps_usage_corr:.3f}**")

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=m_d_filtered, x='apps_installed', y='screen_on_hours',
                hue='gender', alpha=0.6, ax=ax3)
ax3.set_xlabel('Number of Apps Installed')
ax3.set_ylabel('Screen On Time (hours/day)')
ax3.text(0.05, 0.95, f'Correlation: {apps_usage_corr:.3f}',
         transform=ax3.transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
st.pyplot(fig3)

# -----------------------
# Age Analysis
# -----------------------
st.subheader("👥 Age-Based Analysis")

fig_age, (ax_age1, ax_age2) = plt.subplots(1, 2, figsize=(15, 6))

# Age vs App Usage
sns.scatterplot(data=m_d_filtered, x='age', y='app_usage_min', hue='gender', alpha=0.6, ax=ax_age1)
ax_age1.set_xlabel('Age')
ax_age1.set_ylabel('App Usage Time (min/day)')
ax_age1.set_title('Age vs App Usage Time')

# Age vs Screen Time
sns.boxplot(data=m_d_filtered, x='age', y='screen_on_hours', ax=ax_age2)
ax_age2.set_xlabel('Age')
ax_age2.set_ylabel('Screen On Time (hours/day)')
ax_age2.set_title('Screen Time Distribution by Age')

plt.tight_layout()
st.pyplot(fig_age)

# -----------------------
# Battery & Screen Usage Analysis
# -----------------------
st.subheader("🔋 Battery Drain by OS")
battery_by_os = m_d_filtered.groupby('os')['battery_drain'].agg(['mean', 'std', 'count'])
st.dataframe(battery_by_os)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=m_d_filtered, x='os', y='battery_drain', ax=ax4)
ax4.set_title('Battery Drain by Operating System')
st.pyplot(fig4)

st.subheader("⌚ Screen On Time by Gender")
screen_time_by_gender = m_d_filtered.groupby('gender')['screen_on_hours'].agg(['mean', 'std', 'count'])
st.dataframe(screen_time_by_gender)

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=m_d_filtered, x='gender', y='screen_on_hours', ax=ax5)
st.pyplot(fig5)

# -----------------------
# Gender Comparison
# -----------------------
st.subheader("⚧️ Gender Comparison")

col_gen1, col_gen2 = st.columns(2)

with col_gen1:
    gender_usage = m_d_filtered.groupby('gender')['app_usage_min'].mean()
    male_usage = gender_usage.get('Male', 0)
    female_usage = gender_usage.get('Female', 0)
    st.metric("Average App Usage - Male", f"{male_usage:.1f} min")
    st.metric("Average App Usage - Female", f"{female_usage:.1f} min")

with col_gen2:
    gender_screen = m_d_filtered.groupby('gender')['screen_on_hours'].mean()
    male_screen = gender_screen.get('Male', 0)
    female_screen = gender_screen.get('Female', 0)
    st.metric("Average Screen Time - Male", f"{male_screen:.1f} hours")
    st.metric("Average Screen Time - Female", f"{female_screen:.1f} hours")

# -----------------------
# Apps Installed Analysis
# -----------------------
st.subheader("📲 Apps Installed Analysis")

fig_apps, (ax_apps1, ax_apps2) = plt.subplots(1, 2, figsize=(15, 6))

# Distribution of Apps Installed
sns.histplot(m_d_filtered['apps_installed'], kde=True, ax=ax_apps1, bins=20)
ax_apps1.axvline(m_d_filtered['apps_installed'].mean(), color='red', linestyle='--', 
                label=f'Mean: {m_d_filtered["apps_installed"].mean():.1f}')
ax_apps1.set_xlabel('Number of Apps Installed')
ax_apps1.set_ylabel('Frequency')
ax_apps1.legend()

# Apps Installed by OS
sns.boxplot(data=m_d_filtered, x='gender', y='apps_installed', ax=ax_apps2)
ax_apps2.set_xlabel('Gender')
ax_apps2.set_ylabel('Number of Apps Installed')

plt.tight_layout()
st.pyplot(fig_apps)
# -----------------------
# Device Models by Gender
# -----------------------
st.subheader("📱 Device Models by Gender")
top_devices = m_d_filtered['device_model'].value_counts().head(10).index
m_d_top_devices = m_d_filtered[m_d_filtered['device_model'].isin(top_devices)]
device_gender_ct = pd.crosstab(m_d_top_devices['device_model'], m_d_top_devices['gender'], normalize='index') * 100
st.dataframe(device_gender_ct.round(2))

fig8, ax8 = plt.subplots(figsize=(12, 8))
device_gender_ct.plot(kind='bar', stacked=True, ax=ax8)
ax8.set_ylabel('Percentage (%)')
ax8.set_xlabel('Device Model')
ax8.set_title('Gender Distribution by Top Device Models')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig8)
# -----------------------
# Correlation Heatmap
# -----------------------
st.subheader("🧮 Correlation Matrix")
numeric_cols = ['app_usage_min', 'screen_on_hours', 'battery_drain', 'apps_installed', 'data_usage_mb', 'age']
correlation_matrix = m_d_filtered[numeric_cols].corr()

fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True,
            fmt='.2f', cbar_kws={'shrink': .8}, ax=ax6)
st.pyplot(fig6)

# -----------------------
# K-Means Clustering
# -----------------------
st.subheader("🤖 K-Means Clustering: App Usage vs Screen Time")
X = m_d_filtered[['app_usage_min', 'screen_on_hours']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters (k):", 2, 10, 5)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
m_d_filtered['cluster'] = kmeans.fit_predict(X_scaled)

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

fig7, ax7 = plt.subplots(figsize=(10, 8))
for i in range(k):
    cluster_data = m_d_filtered[m_d_filtered['cluster'] == i]
    ax7.scatter(cluster_data['app_usage_min'], cluster_data['screen_on_hours'],
                s=40, alpha=0.7, label=f'User Behavior {i}')
ax7.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200,
            edgecolor='white', linewidths=2, label='Cluster Centers')
ax7.set_xlabel('App Usage Time (min/day)')
ax7.set_ylabel('Screen On Time (hours/day)')
ax7.set_title(f'K-Means Clustering (k={k})')
ax7.legend()
ax7.grid(True)
st.pyplot(fig7)
