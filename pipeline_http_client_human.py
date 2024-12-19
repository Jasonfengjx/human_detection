# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 您只能按照许可证的规定使用该文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按照许可证分发的软件按“原样”提供，
# 没有任何明示或暗示的保证或条件。
# 有关许可证下权限和限制的特定语言，请参阅许可证。

import numpy as np
import requests
import json
import base64
import re

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import cv2
import os
import argparse
from datetime import timedelta
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import requests
from fastapi import APIRouter, HTTPException
from pathlib import Path
from aiofiles import open as aio_open
from pydantic import BaseModel

def str2bool(v):
    return v.lower() in ("true", "t", "1")

parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument("--image_dir", type=str, default="/home/fengjiuxin/OCR/PaddleOCR/deploy/video/file/2/pics")
parser.add_argument("--det", type=str2bool, default=True)
parser.add_argument("--rec", type=str2bool, default=True)
args = parser.parse_args()

def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif'}
    return any([path.lower().endswith(e) for e in img_end])
def milliseconds_to_timestamp(ms):
    """
    将毫秒转换为时间戳格式 HH:MM:SS.mmm

    :param ms: 毫秒数
    :return: 格式化的时间戳字符串
    """
    td = timedelta(milliseconds=ms)
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = ms % 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"

def extract_number_before_jpg(file_path):
    
    pattern = r'(\d+)ms\.jpg$'
    
    match = re.search(pattern, file_path)
    
    if match:
        number_str = match.group(1)
        return str(milliseconds_to_timestamp(int(number_str)))
    else:
        # 如果没有匹配到，返回None
        return None
    
test_img_dir = args.image_dir
def imgs_to_json(test_img_dir, model=1,vedio_path = '/home/fengjiuxin/OCR/PaddleOCR/deploy/video/data/yiche1.mp4.mp4'):
    if model == 1:
        model_id = 'damo/cv_tinynas_human-detection_damoyolo'
        human_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
    else:
        human_detection =  pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')
    
    test_img_list = []
    if os.path.isfile(test_img_dir) and _check_image_file(test_img_dir):
        test_img_list.append(test_img_dir)
    elif os.path.isdir(test_img_dir):
        for single_file in os.listdir(test_img_dir):
            file_path = os.path.join(test_img_dir, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                test_img_list.append(file_path)
    if len(test_img_list) == 0:
        raise Exception("not found any img file in {}".format(test_img_dir))
    if model == 1:
        results = []
        for idx, img_file in enumerate(test_img_list):
            result = human_detection(img_file)
            for key in result:
                result[key] = str(result[key])

            results.append({'file':img_file, 'result':result, 'ts':extract_number_before_jpg(img_file)})
    else:
        results = []
        for idx, img_file in enumerate(test_img_list):
            result = human_detection(img_file)
            
            mapping = {
                0: 'person',
                1: 'face',
                2: 'hand',
                '0': 'person',
                '1': 'face',
                '2': 'hand',
            }
            mapped_list = list(map(lambda x: mapping.get(x, 'unknown'), result['labels']))
            result['labels'] = mapped_list
            for key in result:
                result[key] = str(result[key])
            results.append({'file':img_file, 'result':result, 'ts':extract_number_before_jpg(img_file)})
    return results
    

def extract_frames_from_video(file_path):
    output_folder = os.path.join(os.path.dirname(file_path), "pics")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int((frame_count / fps) * 1000)
        frame_name = f"{timestamp_ms}ms.jpg"
        output_path = os.path.join(output_folder, frame_name)

        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_count} at time {timestamp_ms} ms")

        frame_count += 1
        # break
    cap.release()
    return output_folder

def cal_person_in_result(results):
    total = len(results)

    T = 0
    for result in results:
        if 'person' in result['result']['labels']:
            T += 1
    return T / total
app = FastAPI()
 
@app.post("/human/upload")
async def upload(file: UploadFile = File(...)):
    fn = file.filename
    base_path = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save'
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    if subdirs:
        max_subdir = max(subdirs, key=int)
        new_subdir = str(int(max_subdir) + 1)
        save_path = os.path.join(base_path, new_subdir)
    else:
        save_path = os.path.join(base_path, '0')
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    fname = os.path.basename(fn)
    save_file = os.path.join(save_path, fname)

    print(f"save_file: {save_file}")
    f = open(save_file, 'wb')
    data = await file.read()
    f.write(data)
    f.close()
    figs_path = extract_frames_from_video(save_file)
    results = imgs_to_json(figs_path,model=1)
    person_rate = cal_person_in_result(results)
    if person_rate>0.4:
        print(f"person_rate: {person_rate}")
        return {"msg": f'{fn}上传成功', 'length': len(data), 'person_rate':person_rate, 'result': results, }

    results = imgs_to_json(figs_path,model=2)
    person_rate = cal_person_in_result(results)
    print(f"person_rate: {person_rate}")

    return {"msg": f'{fn}上传成功', 'length': len(data),'person_rate':person_rate, 'result': results}

import base64
import os
class UploadFile(BaseModel):
    filename: str
    data: str  # Base64 encoded string
@app.post("/human/upload_base64")
async def upload_base64(file: UploadFile):
    fn = file.filename
    base_path = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save'
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    if subdirs:
        max_subdir = max(subdirs, key=int)
        new_subdir = str(int(max_subdir) + 1)
        save_path = os.path.join(base_path, new_subdir)
    else:
        save_path = os.path.join(base_path, '0')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, fn)
    try:
        # Decode the base64 string
        file_data = base64.b64decode(file.data)
        print(f"save_file: {save_file}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 data: {str(e)}")

    with open(save_file, 'wb') as f:
        f.write(file_data)

    # Continue with your processing logic
    figs_path = extract_frames_from_video(save_file)
    results = imgs_to_json(figs_path, model=1)
    person_rate = cal_person_in_result(results)
    if person_rate > 0.4:
        print(f"person_rate: {person_rate}")
        return {"msg": f'{fn}上传成功', 'length': len(file_data), 'person_rate':person_rate, 'result': results}

    results = imgs_to_json(figs_path, model=2)
    person_rate = cal_person_in_result(results)
    print(f"person_rate: {person_rate}")

    return {"msg": f'{fn}上传成功', 'length': len(file_data), 'person_rate':person_rate, 'result': results}
router = APIRouter()

from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
import httpx
import aiofiles
class VideoUrl(BaseModel):
    url: str
router = APIRouter()
@router.post("/human/upload_url")
async def download_and_process(video_data: VideoUrl):
    url = video_data.url
    print(f"Downloading video from URL: {url}")
    base_path = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save'
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    if subdirs:
        max_subdir = max(subdirs, key=int)
        new_subdir = str(int(max_subdir) + 1)
        save_path = os.path.join(base_path, new_subdir)
    else:
        save_path = os.path.join(base_path, '0')
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Ensures creation of needed directories

    fname = Path(url.split('/')[-1]).name
    save_file = os.path.join(save_path, fname)

    try:
        # Asynchronously download the file using httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()  # Raises exception for HTTP errors

        # Asynchronously save the downloaded content
        async with aiofiles.open(save_file, 'wb') as f:
            await f.write(response.content)

        # Assume the following functions are correctly defined to handle video processing
        figs_path = extract_frames_from_video(save_file)
        results = imgs_to_json(figs_path, model=1)
        person_rate = cal_person_in_result(results)
        if person_rate > 0.4:
            print(f"person_rate: {person_rate}")
            return {"msg": f'{fname}上传成功', 'length': len(response.content), 'person_rate':person_rate, 'result': results}

        results = imgs_to_json(figs_path, model=2)
        person_rate = cal_person_in_result(results)
        print(f"person_rate: {person_rate}")

        return {"msg": f'{fname}上传成功', 'length': len(response.content),'person_rate':person_rate, 'result': results}
    except httpx.HTTPStatusError as e:
        # Handle HTTP status errors that occur during the download process
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {str(e)}")
    except httpx.RequestError as e:
        # Handle any errors that occur during the connection or file write process
        raise HTTPException(status_code=400, detail=f"Request error: {str(e)}")
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=30002)
    # imgs_to_json('/home/fengjiuxin/OCR/Detect_person/test/15/pics')
    # print()
    # upload('/home/fengjiuxin/OCR/Video_clip/data/2.mp4')
    # print()