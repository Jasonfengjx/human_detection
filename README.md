# Human detect

## Introduction

在准备大型模型的短视频语料时，容易侵犯个人肖像权。为此，我们开发了一种实用的人脸检测技术，用于筛选出不包含人物的短视频。通过对大量现有检测方法的实验，我们筛选出两种互补的方法，并采用自定义规则进行模型集成，成功实现了预期目标。

# 流程

输入：提供了三种输入，分别为视频文件、在线视频链接（内网）、视频base64格式

抽帧：输入的视频首先经过抽帧算法，因为我们对于线上要求是提高视频用户图片的召回率，因而采用密集抽帧方法以保证算法的可用性。如果追求效率和准确率，也可以采用其他的抽帧算法。

模型：damo/cv_tinynas_human-detection_damoyolo、damo/cv_nanodet_face-human-hand-detection

集成：damo/cv_tinynas_human-detection_damoyolo算法可以对大部分人像识别，但是对于密闭空间中化妆女生（目测应该是训练数据集中的问题，训练数据集中缺少化妆亚洲女性这一类别导致了模型的问题）识别率为0，因而采用damo/cv_nanodet_face-human-hand-detection进行补充，具体方式为：统计damo/cv_tinynas_human-detection_damoyolo识别人像的结果，若damo/cv_tinynas_human-detection_damoyolo检测出的比例小于一个固定阈值，则采用damo/cv_nanodet_face-human-hand-detection的结果。

输出：使用fastapi的方式请求，格式为json格式。示例格式如：

``` 
{'msg': '/home/fengjiuxin/OCR/video_data/2.mp4上传成功', 'length': 8179961, 'person_rate': 1.0, 'result': [{'file': '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/2/pics/13700ms.jpg', 'result': {'labels': "['person', 'face']", 'boxes': '[[0, 468, 1016, 1918], [654, 546, 853, 830]]', 'scores': '[0.6863754987716675, 0.8334633111953735]'}, 'ts': '00:00:13.700'}]}
```



效果：使用单独的A算法可以完成大约百分之60的视频数据的人像检测，使用A算法与B算法集成可以完成90%的数据的人像检测

后续操作：剔除有人像的帧

## Install

```shell
conda create -n human_detection python=3.10
conda install mamba
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install modelscope fastapi uvicorn opencv-python addict packaging datasets==2.16.0  oss2 aiofiles httpx python-multipart simplejson sortedcontainers matplotlib thop timm
pip install timm omegaconf numpy opencv-python loguru scikit-image tqdm Pillow thop tabulate easydict

# 修改对应ip与port
nohup python -u pipeline_http_client_human.py > human_detection.log 2>&1 &
# 请求
python post_detect.py
```

# 模型部署加速（todo)

* ONNX

* torch.complie

* openvino

* jit.trace

# 模型原理

使用到的模型的原理：[DAMO-YOLO：一种平衡速度和准确性的新目标检测框架_damoyolo-CSDN博客](https://blog.csdn.net/weixin_42010722/article/details/131392026?ops_request_misc=%7B%22request%5Fid%22%3A%224372be0c9147e9bc871d6c7cce69ed02%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=4372be0c9147e9bc871d6c7cce69ed02&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-131392026-null-null.142^v100^pc_search_result_base8&utm_term=damoyolo&spm=1018.2226.3001.4187)

其他有趣的项目：[ZeYiLin/ThumbnailRecommend: 基于人脸检测与表情识别的短视频封面推荐算法](https://swanhub.co/ZeYiLin/ThumbnailRecommend)