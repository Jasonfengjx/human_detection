from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
input_location = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/1/pics/0ms.jpg'

face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')
result_status = face_human_hand_detection('/home/fengjiuxin/OCR/Detect_person/test/15/pics/0ms.jpg')
# labels = result_status[OutputKeys.LABELS]
# boxes = result_status[OutputKeys.BOXES]
# scores = result_status[OutputKeys.SCORES]
# print("labels is : ", labels)
# print("boxes is : ", boxes)
# print("scores is : ", scores)
results = []
images_dir = '/home/fengjiuxin/OCR/Detect_person/video_data/upload_save/15/pics'
total = len(os.listdir(images_dir))
T, F = 0, 0
for img in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img)
    result = face_human_hand_detection(img_path)
    mapping = {
        0: 'person',
        1: 'face',
        2: 'hand'
    }

    mapped_list = list(map(lambda x: mapping.get(x, 'unknown'), result['labels']))
    result['labels'] = mapped_list
    
    results.append(result)
    print("results is : ", results)
    if len(result['scores'])==0:
        F += 1
        print("no human detected in")
    else:
        T+=1
        print("result is : ", result)
print("total is : ", total)
print('acc:', T/total)
print("results is : ", results)