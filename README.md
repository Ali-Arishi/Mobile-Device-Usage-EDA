📱 Mobile Device Usage and User Behavior Analysis (EDA)

This repository presents a comprehensive Exploratory Data Analysis (EDA) and K-Means clustering model on a dataset describing how users interact with their mobile devices.

🎯 Project Objective

The primary goal of this project is to analyze mobile usage patterns and extract actionable insights that can support:

Improving device performance

Developing effective digital well-being strategies

💾 Dataset Overview

Dataset Name: Mobile Device Usage and User Behavior Dataset

Source: Kaggle

Size: 700 users, 10 features

Key Columns

app_usage_min — Daily app usage (minutes)

screen_on_hours — Daily screen-on time (hours)

battery_drain — Battery consumption (mAh)

data_usage_mb — Mobile data usage (MB/day)

age — User age (18–59)

gender — Male/Female

🛠️ Analysis Methodology
1. Data Cleaning & Preprocessing

Removed missing values and duplicate records

Standardized column names using snake_case

Detected and handled outliers using the Interquartile Range (IQR) method

2. Exploratory Data Analysis (EDA)

Univariate Analysis: Distribution of app time, screen time, battery drain, etc.

Bivariate Analysis:

Strong relationship between app usage and screen-on hours

Comparison of behavior across genders

Correlation Heatmap: Identified numerical variable relationships

3. Clustering (Unsupervised Learning)

Applied K-Means with k = 5 to segment users into distinct behavioral groups

💡 Key Findings and Insights

Strong Correlation:
App usage and screen-on time show a very strong positive correlation (ρ ≈ 0.95).

Minimal Gender Differences:
Usage patterns between genders show only slight variation.

Age Has Minimal Impact:
Age has a weak correlation with the number of installed apps (r ≈ 0).

User Segmentation:
Five unique user groups were identified, ranging from Light Users to Heavy Users.

Battery Performance:
Battery drain is strongly driven by screen-on time.
Android devices show wider battery consumption variability compared to iOS.

🚀 Future Recommendations

To enhance insights, future work may include:

Granular App Categories:
Examining usage based on social, productivity, gaming, etc.

Time Series Analysis:
Tracking user behavior over weeks or months to uncover long-term patterns.

💻 Technologies Used

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn (K-Means, StandardScaler)

Streamlit (Interactive Dashboard)
