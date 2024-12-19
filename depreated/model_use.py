from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
model_id = 'damo/cv_tinynas_human-detection_damoyolo'
input_location = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/1/pics/0ms.jpg'

human_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
result = human_detection(input_location)
print("result is : ", result)

images_dir = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/15/pics'
total = len(os.listdir(images_dir))
T, F = 0, 0
for img in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img)
    result = human_detection(img_path)
    if len(result['scores'])==0:
        F += 1
        print("no human detected in")
    else:
        T+=1
        print("result is : ", result)
print("total is : ", total)
print('acc:', T/total)