#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
df = pd.read_csv('ChinaIncome.csv')
df


# In[3]:


import pandas as pd
input_csv_file = 'ChinaIncome.csv'
output_csv_file = 'output_csv_file'
df = pd.read_csv(input_csv_file)
cleaned_df = df.drop_duplicates()
cleaned_df.to_csv(output_csv_file, index=False)
print("Duplicate values removed and saved to", output_csv_file)


# In[4]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
df = pd.read_csv('output_csv_file')
df


# In[5]:


import pandas as pd
original_df = pd.read_csv('ChinaIncome.csv')
cleaned_df = pd.read_csv('output_csv_file')
print("Original file shape:", original_df.shape)
print("Cleaned file shape:", cleaned_df.shape)
difference_rows = original_df[~original_df.isin(cleaned_df)].dropna()
print("Rows in original file but not in cleaned file:")
print(difference_rows)
difference_columns = original_df[~original_df['agriculture'].isin(cleaned_df['agriculture'])]
print("Rows in original file with ColumnA values not present in cleaned file:")
print(difference_columns)
are_equal = original_df.equals(cleaned_df)
print("Are the two DataFrames equal?", are_equal)


# In[6]:


import pandas as pd
df = pd.read_csv('ChinaIncome.csv')
print("Original DataFrame:")
print(df.head())
df_cleaned = df.drop_duplicates()
df_cleaned['new_feature'] = df_cleaned['agriculture'] * df_cleaned['commerce']
print("Missing values after cleaning:")
print(df_cleaned.isnull().sum())
df_cleaned.to_csv('cleaned_data.csv', index=False)

print("Data wrangling complete. Cleaned data saved to 'cleaned_data.csv'.")


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('cleaned_data.csv')
print("Dataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nFirst few rows of the DataFrame:")
print(df.head())
#Histogram of a numerical column
plt.figure(figsize=(8, 6))
plt.hist(df['agriculture'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Numerical Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Box plot of a numerical column
plt.figure(figsize=(8, 6))
plt.boxplot(df['commerce'])
plt.title('Box plot of Numerical Column')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Scatter plot of two numerical columns
plt.figure(figsize=(8, 6))
plt.scatter(df['construction'], df['commerce'], color='green', alpha=0.5)
plt.title('Scatter plot of Numerical Column 1 vs Numerical Column 2')
plt.xlabel('Numerical Column 1')
plt.ylabel('Numerical Column 2')
plt.grid(True)
plt.show()

#Bar chart of a categorical column
plt.figure(figsize=(8, 6))
df['transport'].value_counts().plot(kind='bar', color='orange')
plt.title('Bar Chart of Categorical Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[8]:


dataset_info = df.info()
print("Dataset Information:")
print(dataset_info)
print("\n")
summary_statistics = df.describe()
print("Summary Statistics:")
print(summary_statistics)
print("\n")
first_few_rows = df.head()
print("First few rows of the DataFrame:")
print(first_few_rows)
print("\n")
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)
print("\n")
print("Conclusion:")
print("1. The dataset contains {} rows and {} columns.".format(df.shape[0], df.shape[1]))
print("2. Summary statistics provide insights into the distribution and central tendency of numerical variables.")
print("3. The first few rows of the DataFrame give a glimpse of the actual data and its format.")
print("4. Missing values are present in certain columns and need to be addressed through imputation or removal.")
print("5. Visualizations reveal patterns, relationships, and distributions within the data.")
print("6. Further analysis is warranted to explore correlations, trends, and potential insights that can guide decision-making.")


# In[ ]:





# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# Dictionary to store predictions for each column
predictions_dict = {}

# Iterate through each column (except the target column) to make predictions
for column in df.columns:
    if column != 'target_column':  # Assuming 'target_column' is the column you want to predict
        # Features and target variable
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model (You can replace LinearRegression with any other ML algorithm)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions using the trained model
        predictions = model.predict(X_test)
        
        # Add the predictions to the dictionary with new column names
        new_column_name = f'predicted_{column}'
        predictions_dict[new_column_name] = predictions

        # Evaluate the model's performance (optional)
        test_score = r2_score(y_test, predictions)
        print(f"R^2 Score for {column}: {test_score}")

# Create a DataFrame from the predictions dictionary
predictions_df = pd.DataFrame(predictions_dict)

# Remove the predicted values columns with row names
df_predicted = df.join(predictions_df)

# Print the predicted data for all 37 rows
print("Predicted Data for All 37 Rows:")
print(df_predicted)

# Explanation of prediction process in a neat tabular format
print("\nExplanation of Prediction Process:")
print("--------------------------------------------------------------------------")
print("| Column Name        | Model Used      | R^2 Score     | Prediction Method |")
print("--------------------------------------------------------------------------")
for column in df.columns:
    if column != 'target_column':
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the model's performance
        test_score = r2_score(y_test, model.predict(X_test))
        
        print(f"| {column:<18} | Linear Regression | {test_score:.4f}       | Model used to predict the column {column} |")
print("--------------------------------------------------------------------------")


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# Dictionary to store predictions for each column
predictions_dict = {}

# Iterate through each column (except the target column) to make predictions
for column in df.columns:
    if column != 'target_column':  # Assuming 'target_column' is the column you want to predict
        # Features and target variable
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model (You can replace LinearRegression with any other ML algorithm)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions using the trained model
        predictions = model.predict(X_test)
        
        # Add the predictions to the dictionary with new column names
        new_column_name = f'predicted_{column}'
        predictions_dict[new_column_name] = predictions

        # Evaluate the model's performance (optional)
        test_score = r2_score(y_test, predictions)
        print(f"R^2 Score for {column}: {test_score}")

        # Plot comparison graph
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color='blue')
        plt.title(f'Comparison between Actual and Predicted values for {column}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()

# Create a DataFrame from the predictions dictionary
predictions_df = pd.DataFrame(predictions_dict)

# Remove the predicted values columns with row names
df_predicted = df.join(predictions_df)

# Print the predicted data for all 37 rows
print("Predicted Data for All 37 Rows:")
print(df_predicted)

# Explanation of prediction process in a neat tabular format
print("\nExplanation of Prediction Process:")
print("--------------------------------------------------------------------------")
print("| Column Name        | Model Used      | R^2 Score     | Prediction Method |")
print("--------------------------------------------------------------------------")
for column in df.columns:
    if column != 'target_column':
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the model's performance
        test_score = r2_score(y_test, model.predict(X_test))
        
        print(f"| {column:<18} | Linear Regression | {test_score:.4f}       | Model used to predict the column {column} |")
print("--------------------------------------------------------------------------")


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mpmath

# Load the dataset into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# Handle missing values
df.dropna(inplace=True)  # Drop rows with missing values

# Replace infinite values or values too large for dtype('float64') with NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)  # Drop rows with large or infinite values

# Set precision to float128
mpmath.mp.dps = 34  # Adjust the precision as needed

# Dictionary to store predictions for each column
predictions_dict = {}

# Iterate through each column (except the target column) to make predictions
for column in df.columns:
    if column != 'target_column':  # Assuming 'target_column' is the column you want to predict
        # Features and target variable
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model (You can replace LinearRegression with any other ML algorithm)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions using the trained model
        predictions = model.predict(X_test)
        
        # Convert the data type of predictions to float128
        predictions = np.array([mpmath.mpf(x) for x in predictions])
        
        # Add the predictions to the dictionary with new column names
        new_column_name = f'predicted_{column}'
        predictions_dict[new_column_name] = predictions

        # Evaluate the model's performance (optional)
        test_score = r2_score(y_test, predictions)
        print(f"R^2 Score for {column}: {test_score}")

        # Plot comparison graph (optional)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color='blue')
        plt.title(f'Comparison between Actual and Predicted values for {column}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()

# Create a DataFrame from the predictions dictionary
predictions_df = pd.DataFrame(predictions_dict)

# Remove the predicted values columns with row names
df_predicted = df.join(predictions_df)

# Print the predicted data for all rows
print("Predicted Data for All Rows:")
print(df_predicted)

# Explanation of prediction process in a neat tabular format
print("\nExplanation of Prediction Process:")
print("--------------------------------------------------------------------------")
print("| Column Name        | Model Used      | R^2 Score     | Prediction Method |")
print("--------------------------------------------------------------------------")
for column in df.columns:
    if column != 'target_column':
        X = df.drop(column, axis=1)
        y = df[column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        test_score = r2_score(y_test, model.predict(X_test))
        print(f"| {column:<18} | Linear Regression | {test_score:.4f}       | Model used to predict the column {column} |")
print("--------------------------------------------------------------------------")


# In[ ]:




