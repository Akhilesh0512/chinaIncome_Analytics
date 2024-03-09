import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mpmath
df = pd.read_csv('cleaned_data.csv')
df.dropna(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

mpmath.mp.dps = 34 
predictions_dict = {}
for column in df.columns:
    if column != 'target_column':  
        X = df.drop(column, axis=1)
        y = df[column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = np.array([mpmath.mpf(x) for x in predictions])
        new_column_name = f'predicted_{column}'
        predictions_dict[new_column_name] = predictions
        test_score = r2_score(y_test, predictions)
        print(f"R^2 Score for {column}: {test_score}")
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color='blue')
        plt.title(f'Comparison between Actual and Predicted values for {column}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()
predictions_df = pd.DataFrame(predictions_dict)
df_predicted = df.join(predictions_df)
print("Predicted Data for All Rows:")
print(df_predicted)
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
