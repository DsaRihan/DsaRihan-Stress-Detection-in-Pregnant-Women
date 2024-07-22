


# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score
# import pickle

# # Load data from CSV file
# file_path = 'katenew.csv'  # Update with your file path
# df = pd.read_csv(file_path, dtype={'id': str})

# df.drop(['datetime', 'id'], axis=1, inplace=True)

# # Impute missing values
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(df[['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP']])
# y = df['label']  # Target labels

# # Initialize k-Nearest Neighbors Classifier
# knn = KNeighborsClassifier(n_neighbors=3)

# # Train the classifier on the entire dataset
# knn.fit(X_imputed, y)

# # Save the trained model as a pickle file
# with open('knn_model1.pkl', 'wb') as file:
#     pickle.dump(knn, file)

# # Example of loading and using the trained model from the pickle file
# with open('katenew.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# # Example of predicting label for new data using the loaded model
# new_data = [[-13, -61, 5, 6.769995, 99.43, 31.17]]  # Example new data
# predicted_label = loaded_model.predict(new_data)
# print("Predicted label for new data using the loaded model:", predicted_label[0])

# # Predict labels for the entire dataset
# predicted_labels = loaded_model.predict(X_imputed)

# # Calculate accuracy of the KNN classifier
# accuracy = accuracy_score(y, predicted_labels)
# print("Accuracy of KNN classifier:", accuracy)


import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pickle

# Load data from CSV file
file_path = 'katenew.csv'  # Update with the correct file path
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File not found. Please provide the correct file path.")
    exit()

# Drop unnecessary columns
df.drop(['datetime', 'id'], axis=1, inplace=True)

# Convert 'EDA' column to floats
df['EDA'] = pd.to_numeric(df['EDA'], errors='coerce')

# Initialize a SimpleImputer with a strategy to handle missing values
imputer = SimpleImputer(strategy='mean')

# Impute missing values for all columns
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target variable
X = df.drop('label', axis=1)  # Features
y = df['label']               # Target variable

# Initialize k-Nearest Neighbors Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# Train the regressor on the entire dataset
knn_regressor.fit(X, y)

# Save the trained model as a pickle file
with open('knn_regressor_model.pkl', 'wb') as file:
    pickle.dump(knn_regressor, file)

# Example of loading and using the trained model from the pickle file
with open('knn_regressor_model.pkl', 'rb') as file:
    loaded_regressor = pickle.load(file)

# Example of predicting label for new data using the loaded model

# new_data = [[-31,-78,-15,-39.98,60.02,33.002]]   #label 2
# new_data = [[-23,53,26,-29.09,70.87,34.091]]   #label 1
new_data = [[-1,60,22,-19.22,80.68,35.078]]   #label 0


predicted_label = loaded_regressor.predict(new_data)
print("Predicted label for new data using the loaded model:", predicted_label[0])

# Predict labels for the entire dataset
predicted_labels = loaded_regressor.predict(X)

# Calculate Mean Squared Error (MSE) of the regressor
mse = mean_squared_error(y, predicted_labels)
print("Mean Squared Error (MSE) of KNN regressor:", mse)
from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error (MAE) of the regressor
mae = mean_absolute_error(y, predicted_labels)
print("Mean Absolute Error (MAE) of KNN regressor:", mae)
