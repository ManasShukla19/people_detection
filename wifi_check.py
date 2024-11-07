import requests
from PIL import Image
from io import BytesIO

# Replace with your camera's IP address, username, and password
camera_ip = "10.222.74.239"  # e.g., "192.168.1.10"
username = "boschcamera"
password = "122333"

# Snapshot URL
url = f"http://{camera_ip}/snapshot.jpg"

try:
    # Fetch the snapshot
    response = requests.get(url, auth=(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        # Open the image
        img = Image.open(BytesIO(response.content))
        img.show()  # Display the image
    else:
        print("Failed to retrieve snapshot. Status code:", response.status_code)

except Exception as e:
    print("An error occurred:", e)
