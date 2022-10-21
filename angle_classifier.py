import requests
import json
import base64
from PIL import Image
from datetime import datetime
from pprint import pprint
import os
import shutil

URL = 'http://localhost:8090/car-info/invocations'
IMAGE = '/Users/maixueqiao/Downloads/Project/cropped_images/sample4-cropped/carro_00eipeE4aDG5Npe7.jpg'
ALB_URL = 'http://cv-inspection-alb-117943900.ap-southeast-1.elb.amazonaws.com/car-info/invocations'

if __name__ == "__main__":
    dir = os.listdir('sample3-cropped')
    dst_dir = 'front_rear_images'

    for im in dir:
        if (im.endswith('.jpg')):
            img_str = base64.b64encode(open('sample3-cropped'+'/'+im, 'rb').read()).decode('utf-8')
            data = {'img_str': img_str}

            print('Sending request..')
            start_time = datetime.now()
            response = requests.post(ALB_URL, data=json.dumps(data))
            result = response.json()
            pprint(result['car_angle'])
            if (result['car_angle'] == 'front') or (result['car_angle'] == 'rear') :
                print('yes')
                shutil.copy('sample3-cropped'+'/'+im, dst_dir)
            else:
                print('no')
        
