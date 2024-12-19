# Human Detection

## Introduction

When preparing short video corpora for large models, it is easy to infringe on personal portrait rights. Therefore, we have developed a practical face detection technology to filter out short videos that do not contain people. Through experiments on a large number of existing detection methods, we have selected two complementary methods and adopted custom rules for model integration, successfully achieving the expected goals.

# Process

Input: Three types of input are provided: video files, online video links (intranet), and video in base64 format.

Frame Extraction: The input video first undergoes a frame extraction algorithm. Since we require high recall rates for video user images online, we use a dense frame extraction method to ensure the algorithm's usability. If efficiency and accuracy are pursued, other frame extraction algorithms can also be used.

Model: damo/cv_tinynas_human-detection_damoyolo, damo/cv_nanodet_face-human-hand-detection

Integration: The damo/cv_tinynas_human-detection_damoyolo algorithm can recognize most human images, but it has a 0% recognition rate for made-up women in enclosed spaces (presumably due to the lack of this category in the training dataset). Therefore, we use damo/cv_nanodet_face-human-hand-detection as a supplement. Specifically, we count the results of damo/cv_tinynas_human-detection_damoyolo. If the proportion of recognized human images is less than a fixed threshold, we use the results of damo/cv_nanodet_face-human-hand-detection.

Output: Requests are made using FastAPI, and the format is JSON. An example format is as follows:

``` 
{'msg': '/home/fengjiuxin/OCR/video_data/2.mp4 uploaded successfully', 'length': 8179961, 'person_rate': 1.0, 'result': [{'file': '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/2/pics/13700ms.jpg', 'result': {'labels': "['person', 'face']", 'boxes': '[[0, 468, 1016, 1918], [654, 546, 853, 830]]', 'scores': '[0.6863754987716675, 0.8334633111953735]'}, 'ts': '00:00:13.700'}]}
```

Effect: Using algorithm A alone can complete about 60% of the human image detection in video data. Using the integration of algorithms A and B can complete 90% of the data's human image detection.

Subsequent Operation: Remove frames with human images.

## Install

```shell
conda create -n human_detection python=3.10
conda install mamba
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install modelscope fastapi uvicorn opencv-python addict packaging datasets==2.16.0  oss2 aiofiles httpx python-multipart simplejson sortedcontainers matplotlib thop timm
pip install timm omegaconf numpy opencv-python loguru scikit-image tqdm Pillow thop tabulate easydict

# Modify the corresponding IP and port
nohup python -u pipeline_http_client_human.py > human_detection.log 2>&1 &
# Request
python post_detect.py
```

# Model Deployment Acceleration (todo)

* ONNX

* torch.compile

* openvino

* jit.trace

# Model Principle

The principle of the model used: [DAMO-YOLO: A New Object Detection Framework Balancing Speed and Accuracy_damoyolo-CSDN Blog](https://blog.csdn.net/weixin_42010722/article/details/131392026?ops_request_misc=%7B%22request%5Fid%22%3A%224372be0c9147e9bc871d6c7cce69ed02%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=4372be0c9147e9bc871d6c7cce69ed02&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-131392026-null-null.142^v100^pc_search_result_base8&utm_term=damoyolo&spm=1018.2226.3001.4187)

Other interesting projects: [ZeYiLin/ThumbnailRecommend: Short Video Cover Recommendation Algorithm Based on Face Detection and Expression Recognition](https://swanhub.co/ZeYiLin/ThumbnailRecommend)
