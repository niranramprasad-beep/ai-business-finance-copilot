# Day 2: Superstore business questions
# Run with: python3 superstore_analysis.py
# BEFORE RUNNING: download the Superstore Sales dataset from Kaggle,
# put the CSV in this folder, and update the filename below if it's different.

import pandas as pd

# 1. Load the CSV into a DataFrame (a table)
df = pd.read_csv("superstore.csv")  # <-- change name if your file is called something else

# 2. Look at what you're working with
print(df.head())          # first 5 rows
print(df.columns.tolist())  # all column names — check these match the ones used below!

# NOTE: column names below ("Sales", "Product Name", "Region") are the usual
# Kaggle Superstore names. If print(df.columns) shows different names, swap them in.

# 3. Question 1: total revenue
total_revenue = df["Sales"].sum()
print(f"Total revenue: ${total_revenue:,.2f}")

# 4. Question 2: top 5 products by revenue
top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(5)
print("\nTop 5 products by revenue:")
print(top_products)

# 5. Question 3: revenue by region
revenue_by_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print("\nRevenue by region:")
print(revenue_by_region)
