from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

from modelscope.pipelines import pipeline


human_detection = pipeline('domain-specific-object-detection', 'IoT-Edge/Head_Person_Detection', model_revision='v1.0.0')


images_dir = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/15/pics'
total = len(os.listdir(images_dir))
T, F = 0, 0
for img in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img)
    result = human_detection(img_path)
    print("result is : ", result)
    if not result:
        F += 1
        continue
    if len(result['scores'])==0:
        F += 1
        print("no human detected in")
    else:
        T+=1
        print("result is : ", result)
print("total is : ", total)
print('acc:', T/total)