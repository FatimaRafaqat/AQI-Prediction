import pandas as pd

# Load the dataset (update path if needed)
df = pd.read_csv("C:\\Users\\PMLS\\Desktop\\AQI-Prediction\\aqi_result.csv")

# View basic info
print(df.shape)
print(df.columns)
print(df.head())

################Step 2: Summary Statistics######################
# General statistics
print(df.describe())

# Check data types
print(df.dtypes)


################Step 3: Missing Values###########################
print("For Null Values:")
# Count missing values
print(df.isnull().sum())

# Visualize missing values
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd")
plt.title("Missing Values Heatmap")
plt.show()


################Step 4: Distribution of Each Feature###########################
numeric_cols = df.select_dtypes(include='number').columns
print(df[numeric_cols].describe().T)

################Step 5: Correlation Heatmap###########################

plt.figure(figsize=(10, 8))

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Plot correlation heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

################Step 6: Time Series Trend of AQI ######################

df['timestamp_str'] = pd.to_datetime(df['timestamp_str'])

plt.figure(figsize=(14,6))
plt.plot(df['timestamp_str'], df['calculated_aqi'], marker='o')
plt.title("AQI Over Time")
plt.xlabel("Timestamp")
plt.ylabel("AQI")
plt.grid()
plt.show()


################Step 7: Outlier Detection (Boxplots) ######################
for col in ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3','no']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outlier detection for {col}")
    plt.show()
