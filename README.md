# 📱 Mobile Device Usage and User Behavior Analysis (EDA)

This repository presents a comprehensive Exploratory Data Analysis (EDA) and K-Means clustering model on a dataset describing user behavior and usage patterns of mobile devices.

---

## 🎯 Project Objective

The main goal of this project is to analyze mobile usage patterns and extract actionable insights to:

- Improve device performance  
- Develop digital well-being strategies  

---

## 💾 Dataset Overview

- **Dataset Name:** Mobile Device Usage and User Behavior Dataset  
- **Source:** Kaggle  
- **Size:** 700 users, 10 features  

### Key Columns
- `app_usage_min` — Daily app usage (minutes)  
- `screen_on_hours` — Daily screen-on time (hours)  
- `battery_drain` — Daily battery consumption (mAh)  
- `data_usage_mb` — Daily mobile data usage (MB)  
- `age` — User age (18–59)  
- `gender` — User gender (Male/Female)  

---

## 🛠️ Analysis Methodology

### **Data Cleaning & Preprocessing**
- Removed missing values and duplicate entries  
- Standardized column names (snake_case)  
- Detected and handled outliers using the Interquartile Range (IQR) method  

### **Exploratory Data Analysis (EDA)**
- **Univariate Analysis:** Examined the distribution of key usage metrics  
- **Bivariate Analysis:**  
  - Relationship between app usage and screen-on hours  
  - Comparison across genders  
- **Correlation Analysis:** Visualized relationships among numerical variables  

### **Clustering**
- Applied **K-Means Clustering** with `k=5` to segment users into distinct behavioral groups  

---

## 💡 Key Findings

- **Strong Correlation:** App usage and screen-on time show a strong positive correlation (ρ ≈ 0.95)  
- **Minimal Gender Differences:** Usage patterns are similar across genders  
- **Age Impact:** Weak correlation with the number of installed apps (r ≈ 0)  
- **User Segments:** Identified five distinct groups from *Light Users* to *Heavy Users*  
- **Battery Performance:** Battery drain is strongly driven by screen-on time; Android devices show greater variability than iOS  

---

## 🚀 Future Recommendations

- **Granular App Categories:** Include app-specific usage (e.g., Social Media, Productivity)  
- **Time Series Analysis:** Examine usage patterns over weeks or months  

---

## 💻 Technologies Used

- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn (K-Means, StandardScaler)  
- Streamlit (Interactive Dashboard)  
