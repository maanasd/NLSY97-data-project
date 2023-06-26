import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score

pd.options.display.float_format = '{:,.2f}'.format

# Check if the cleaned file exists
if os.path.isfile('cleaned_file.csv'):
    # Read the cleaned data file
    df_cleaned = pd.read_csv('cleaned_file.csv')
else:
    # Read the original data file
    df_data = pd.read_csv('NLSY97_subset.csv')

    # Data Cleaning - Check for duplicates
    df_cleaned = df_data.drop_duplicates()

    # Save the cleaned DataFrame to a new file
    df_cleaned.to_csv('cleaned_file.csv', index=False)

df_cleaned = df_cleaned.rename(columns={'S': 'Years of Schooling', 'EXP': 'Experience', 'EARNINGS': 'Earnings'})
statistics = df_cleaned.describe()
print(statistics)

# Seaborn pairplot
sns.set(style='ticks', font_scale=1.2)
sns.pairplot(df_cleaned[['Years of Schooling', 'Experience', 'Earnings']])
plt.show()

# Plotly scatter plot
fig = px.scatter(df_cleaned, x='Years of Schooling', y='Earnings',
                 title='Relationship between Years of Schooling and Earnings',
                 trendline='ols')
fig.update_traces(marker=dict(size=5, color='steelblue'))
fig.update_layout(title_font=dict(size=20))
fig.show()

from sklearn.model_selection import train_test_split

# Split the dataset into training and test datasets
X = df_cleaned[['Years of Schooling']]  # Input feature
y = df_cleaned['Earnings']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the earnings for the training data
y_train_pred = model.predict(X_train)

# Calculate the R-squared score for the training data
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (training):", r2_train)

# Get the coefficients of the model
coef = model.coef_

# Print the coefficient value
print("Coefficient:", coef[0])

# Plot predicted values vs. actual values
plt.scatter(X_train, y_train, color='blue', label='Actual')
plt.scatter(X_train, y_train_pred, color='red', label='Predicted')
plt.xlabel('Years of Schooling')
plt.ylabel('Earnings')
plt.legend()
plt.show()

# Prepare the input features and target variable
X = df_cleaned[['Years of Schooling', 'Experience']]  # Input features
y = df_cleaned['Earnings']  # Target variable

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the earnings for the training data
y_train_pred = model.predict(X_train)

# Calculate the R-squared score for the training data
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (training):", r2_train)

# Get the coefficients of the model
coef = model.coef_

# Print the coefficients
for i, feature in enumerate(['Years of Schooling', 'Experience']):
    print("Coefficient for", feature + ":", coef[i])

# Make a prediction for a specific set of input values
prediction_data = {'Years of Schooling': [16], 'Experience': [5]}
prediction_data_reshaped = pd.DataFrame(prediction_data)
prediction = model.predict(prediction_data_reshaped)
print("Predicted earnings:", prediction[0])
