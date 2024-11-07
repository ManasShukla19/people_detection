import requests
import time
# from demo_code import *
from new_v8 import *

# Replace with your Write API Key
API_KEY = 'F6EQ31NYV76I9CRK'

# Replace with your channel field (e.g., field1, field2, etc.)
FIELD = 'count'
while True:
    def main():
        #time.sleep(2)
        result =people_count()        
        return result

    value=main()
    # The integer value you want to send
    # Construct the URL to send the data
    url = f'https://api.thingspeak.com/update?api_key=0OXRZCK0JAT8P9GG&field1='+str(value)

    # Send the HTTP GET request
    response = requests.get(url)

    # Check the response
    if response.status_code == 200:
        print('Data successfully sent to ThinkSpeak.')
    else:
        print('Failed to send data to ThinkSpeak. Response:', response.status_code)
        
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        print("Quitting.............")
        break

if __name__=="__main__":
    main()
