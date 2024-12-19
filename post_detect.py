import requests

import os
import cv2

def upload_video(file_path):
    url = "http://10.168.19.15:30002/human/upload/"
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file, "video/mp4")}
        response = requests.post(url, files=files)
    return response.json()

import requests
import base64
import os
import json

def upload_video_base64(file_path):
    url = "http://10.168.19.15:30002/human/upload_base64"
    with open(file_path, "rb") as file:
        base64_encoded_data = base64.b64encode(file.read()).decode('utf-8')
    data = {
        "filename": os.path.basename(file_path),
        "data": base64_encoded_data
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    return response.json() 

import requests
import json

def send_video_url(url_to_send):
    api_url = "http://10.168.19.15:30002/human/upload_url"
    data = {
        "url": url_to_send
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(api_url, data=json.dumps(data), headers=headers)
    return response.json()

if __name__ == "__main__":

    # 测试MP4
    video_path = "/home/fengjiuxin/OCR/video_data/2.mp4"
    response = upload_video(video_path)
    print(response)
    print(response['person_rate'])
    # 打印并保存response
    with open('response.txt', 'w') as f:
        # 写入response的内容
        f.write(str(response) + '\n')
        
        # 如果需要打印其中的 'person_rate' 信息
        f.write('person_rate: ' + str(response.get('person_rate', 'N/A')) + '\n')
    # 测试base64
    video_path = "/home/fengjiuxin/OCR/video_data/2.mp4"
    response = upload_video_base64(video_path)
    print(response['person_rate'])

    # 测试内网视频
    video_url = "https://qvod.yiche.com/03bd754cvodtransbj1251489075/75543d981397757886426361079/v.f1160447.mp4"
    response = send_video_url(video_url)
    print(response['person_rate'])
