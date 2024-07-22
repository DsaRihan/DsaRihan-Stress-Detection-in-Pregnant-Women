# import numpy as np
# import pickle

# # Load the trained KNN model from the pickle file
# with open('python/knn_model1.pkl', 'rb') as file:
#     knn_model = pickle.load(file)

# # Example of user-defined test data
# test_data = np.array([[-13, -61, 5, 6.769995, 99.43, 31.17],  # Example test data point 1
#                       [-20, -69, -3, 6.769995, 99.43, 31.17],  # Example test data point 2
#                       [-31, -78, -15, 6.769995, 99.43, 31.17]])  # Example test data point 3

# # Predict labels for the test data
# predicted_labels = knn_model.predict(test_data)

# # Display the predicted labels for each test data point
# for i, label in enumerate(predicted_labels):
#     print(f"Test data point {i+1} belongs to label: {label}")



# import socket
# import numpy as np
# import pickle

# # Define the IP address and port to connect to
# HOST = '192.168.4.1'  # Adjust the IP address based on your server
# PORT = 80  # Adjust the port based on your server

# # Load the trained KNN model from the pickle file
# with open('knn_model1.pkl', 'rb') as file:
#     knn_model = pickle.load(file)

# # Create empty lists to store data
# X_list, Y_list, Z_list, EDA_list, HR_list, TEMP_list, id_list, datetime_list = [], [], [], [], [], [], [], []

# # Create a socket object
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     try:
#         # Connect to the server
#         s.connect((HOST, PORT))
#         print('Connected to server')

#         # Receive data from the server
#         while True:
#             data = s.recv(1024)  # Adjust the buffer size as needed
#             if not data:
#                 break
#             received_text = data.decode('utf-8')
#             print('Received data from Wi-Fi:', received_text)
            
#             # Split the received data into individual values
#             values = received_text.strip().split(',')
#             if len(values) == 8:
#                 X, Y, Z, EDA, HR, TEMP, id_, datetime_ = map(float, values)
                
#                 # Store data in individual variables
#                 X_list.append(X)
#                 Y_list.append(Y)
#                 Z_list.append(Z)
#                 EDA_list.append(EDA)
#                 HR_list.append(HR)
#                 TEMP_list.append(TEMP)
#                 id_list.append(int(id_))
#                 datetime_list.append(datetime_)
                
#                 # Create test data from received sensor data
#                 test_data = np.array([[X, Y, Z, EDA, HR, TEMP]])
                
#                 # Predict label for the test data
#                 predicted_label = knn_model.predict(test_data)
#                 print(f"Predicted label for data point with id {id_}: {predicted_label[0]}")
                
#     except ConnectionRefusedError:
#         print('Connection refused. Please check the server IP address and port.')
#     except Exception as e:
#         print('An error occurred:', e)
#     finally:
#         # Close the socket
#         s.close()
#         print('Connection closed')




# import numpy as np
# import pickle

# # Load the trained KNN model from the pickle file
# with open('knn_model1.pkl', 'rb') as file:
#     knn_model = pickle.load(file)

# # Example of user-defined test data
# test_data = np.array([[-13, -61, 5, 6.769995, 99.43, 31.17]])  # Example test data point as one row

# # Extract individual values from the test data
# x, y, z, eda, heartbeat, timestamp = test_data[0][0:]

# # Predict label for the test data
# predicted_label = knn_model.predict([[x, y, z, eda, heartbeat, timestamp]])

# # Display the predicted label
# print(f"Predicted label for the test data: {predicted_label[0]}")



import json
import numpy as np
import pickle
import datetime

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%I:%M:%S %p")
print("Current Time:", formatted_time)

# Load the trained KNN model from the pickle file
with open('knn_model1.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Example of user-defined test data
# test_data = np.array([[-31, -78, -15, 6.769995, 99.43, 31.17]])  # Example test data point as one row
test_data = np.array([[-13, -61, 5, 6.769995, 99.43, 31.17]])  # Example test data point as one row
# test_data = np.array([[-62.0,3.0,17.0,0.397093,71.58,29.01]])  # Example test data point as one row÷

# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(test_data)


# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)

# Predict label for the test data
predicted_label = knn_model.predict(test_data)

# Create a dictionary to hold the output data
output_data = {
    "label": predicted_label[0],
    "x": test_data[0][0],
    "y": test_data[0][1],
    "z": test_data[0][2],
    "eda": test_data[0][3],
    "heartbeat": test_data[0][4],
    "temperature": test_data[0][5],
    "timestamp":formatted_time
    # "timestamp": test_data[0][7]  # Replace "some_timestamp" with the actual timestamp value
}

# Convert the dictionary to JSON format
output_json = json.dumps(output_data)
with open('data.json', 'w') as file:
    json.dump(output_data, file)

# Print the JSON output
print(output_json)


# import json
# import numpy as np
# import pickle

# # Load the trained AdaBoost model from the pickle file
# with open('qqadaboost_classifier.pkl', 'rb') as file:
#     adaboost_model = pickle.load(file)

# # Example of user-defined test data
# test_data = np.array([[10, 11, 9, 6, 99.042, 36.17]])  # Example test data point as one row

# # Predict label for the test data
# predicted_label = adaboost_model.predict(test_data)

# # Create a dictionary to hold the output data
# output_data = {
#     "label": predicted_label[0],
#     "x": test_data[0][0],
#     "y": test_data[0][1],
#     "z": test_data[0][2],
#     "eda": test_data[0][3],
#     "heartbeat": test_data[0][4],
#     "temperature": test_data[0][5],
#     # "timestamp": test_data[0][7]  # Replace "some_timestamp" with the actual timestamp value
# }

# # Convert the dictionary to JSON format
# output_json = json.dumps(output_data)

# # Save the output JSON to a file
# with open('dataada.json', 'w') as file:
#     json.dump(output_data, file)

# # Print the JSON output
# print(output_json)

# import json
# import numpy as np
# import pickle

# # Load the trained AdaBoost model from the pickle file
# with open('qqadaboost_classifier.pkl', 'rb') as file:
# # with open('adaboost_classifier.pkl', 'rb') as file:
#     adaboost_model = pickle.load(file)

# # Example of user-defined test data
# test_data = np.array([[-13.0,-61.0,5.0,6.769995,99.43,31.17]])  # Example test data point as one row

# # Predict label for the test data
# predicted_label = adaboost_model.predict(test_data)

# # Convert numpy.int64 values to Python integers
# predicted_label = predicted_label.item()
# test_data = test_data.astype(float).tolist()[0]

# # Create a dictionary to hold the output data
# output_data = {
#     "label": predicted_label,
#     "x": test_data[0],
#     "y": test_data[1],
#     "z": test_data[2],
#     "eda": test_data[3],
#     "heartbeat": test_data[4],
#     "temperature": test_data[5],
#     # "timestamp": test_data[0][7]  # Replace "some_timestamp" with the actual timestamp value
# }

# # Convert the dictionary to JSON format
# output_json = json.dumps(output_data)

# # Save the output JSON to a file
# with open('dataada1.json', 'w') as file:
#     json.dump(output_data, file)

# # Print the JSON output
# print(output_json)
