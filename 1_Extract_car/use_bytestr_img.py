import json
import base64
from PIL import Image
import io

if __name__ == '__main__':
    # Read json file (loaded as dict)
    with open('car_only_bytestr.json', 'r') as file:
        data = json.load(file)

    # Extract byte str from json dict
    car_only_base64_image = data['results'][0]['entities'][0]['image']
    
    # Decode base64 str
    image_data = base64.b64decode(car_only_base64_image)

    # Convert binary to img
    image = Image.open(io.BytesIO(image_data))

    # Save the img
    image.save('car_only.png')