# 🏪 Customer Personality Analysis using Clustering

This repository is a project aimed at segmenting a company's customer base into distinct, actionable profiles using various clustering algorithms.

[Notebook on Kaggle](https://www.kaggle.com/code/nikitasem/clustering-customer-personality-analysis)

## What's done

- **Exploratory Data Analysis (EDA):** In-depth analysis of customer demographics, purchasing behavior, and campaign responses.
- **Feature Engineering:** Processing old features, creating new ones and handling multicollinearity using Variance Inflation Factor (VIF).
- **Data Preprocessing Pipeline:** Implementing a robust data pipeline utilizing `StandardScaler` and `KNNImputer` for effective missing value imputation.
- **Model Training & Benchmarking:** Comparing multiple clustering algorithms, including K-Means, Agglomerative Clustering, DBSCAN, MeanShift, Gaussian Mixture, Birch, and Spectral Clustering.
- **Dimensionality Reduction & Visualization:** Applying Principal Component Analysis (PCA) to visualize and evaluate the performance of clustering algorithms on a 2D canvas.
- **Business Insights:** Interpreting the final clusters into three major customer archetypes (Wealthy, Family-Oriented, and Low-Income Customers) to drive personalized marketing strategies.

## Dataset
The Customer Personality Analysis dataset provides detailed profile data of customers, including their spending habits across various categories, demographic details, and company interaction history.

[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?select=marketing_campaign.csv)
