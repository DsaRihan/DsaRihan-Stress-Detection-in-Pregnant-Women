import json
import numpy as np
import pickle
import socket
import datetime
# import random

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%I:%M:%S %p")
# print("Current Time:", formatted_time)


# Define the IP address and port of the device
device_ip = '192.168.4.1'  # Replace with the IP address of your device
device_port = 80  # Replace with the port number your device is listening on

# Initialize the socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the device
    sock.connect((device_ip, device_port))
    print(f"Connected to {device_ip}:{device_port}")

    while True:
        try:
            # Read data from the socket
            serial_data = sock.recv(1024).decode().strip()

            # Check if data is empty
            if not serial_data:
                continue  # Skip processing if data is empty

            # Parse JSON data
            json_data = json.loads(serial_data)
            print("JSON Data:", json_data)
            # p1=np.random.randint(70,75)

            # Extract values from JSON
            D1 = float(json_data["D1"])
            D2 = float(json_data["D2"])
            D3 = float(json_data["D3"])
            D4 = float(json_data["D4"])
            D5 = float(json_data["D5"])
            D6 = float(json_data["D7"])

            # Update test_data with extracted values
            test_data = np.array([[D1, D2, D3, D4, D5, D6]])
            
            print("test_data:", test_data)

            with open('knn_model1.pkl', 'rb') as file:
                knn_model = pickle.load(file)

            # Load the trained KNN model from the pickle file
            # Predict label using the loaded model
            predicted_label = knn_model.predict(test_data)

            # Create a dictionary to hold the output data
            output_data = {
                "label": predicted_label[0],
                "x": D3,
                "y": D4,
                "z": D5,
                "eda": D1,
                "heartbeat":D6,
                "temperature": D2,
                "timestamp":formatted_time
            }


            # Convert the dictionary to JSON format
            with open('data.json', 'w') as file:
                json.dump(output_data, file)

            # Print the JSON output
            print(output_data)

        except json.decoder.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            continue  # Skip processing if JSON decoding fails
        except Exception as e:
            print("Error:", e)

finally:
    # Close the socket connection
    sock.close()
