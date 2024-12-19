from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')
result_status = face_human_hand_detection('/home/fengjiuxin/OCR/Detect_person/test/15/pics/0ms.jpg')
labels = result_status[OutputKeys.LABELS]
boxes = result_status[OutputKeys.BOXES]
scores = result_status[OutputKeys.SCORES]
print("labels is : ", labels)
print("boxes is : ", boxes)
print("scores is : ", scores)