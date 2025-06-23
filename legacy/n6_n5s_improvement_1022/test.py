from dotenv import load_dotenv
import os
import requests
import pickle

load_dotenv()

print(f"Username: {os.getenv('TAIDE_EMAIL')}")
print(f"Password: {os.getenv('TAIDE_PASSWORD')}")

# Define the host URL for the API
host = "https://td.nchc.org.tw/api/v1"

# Get username and password from user input
username = os.getenv('TAIDE_EMAIL')
password = os.getenv('TAIDE_PASSWORD')
print("Username:", username)
print("Password:", password)

# Make a POST request to the API to get the access token
response = requests.post(
    f"{host}/token",
    data={"username": username, "password": password},
    # headers={"Content-Type": "application/x-www-form-urlencoded"}
)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    try:
        # Parse the JSON response and extract the access token
        # token = response.json().get("access_token")
        token = response.json()["access_token"]
        if token:
            print(f"Access Token: {token}")
        else:
            print("Access token not found in the response.")
    except ValueError:
        print("Error parsing the response as JSON.")
else:
    print(f"Failed to retrieve token. Status code: {response.status_code}, Response: {response.text}")
