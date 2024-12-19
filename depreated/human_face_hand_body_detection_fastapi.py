from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uvicorn
import io

app = FastAPI()

# 初始化模型，只需加载一次
model_id = 'damo/cv_tinynas_human-detection_damoyolo'
human_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)

@app.post("/detect_three")
async def detect_human(file_path):
    face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')
    result = face_human_hand_detection(file_path)
    mapping = {
        0: 'person',
        1: 'face',
        2: 'hand'
    }

    mapped_list = list(map(lambda x: mapping.get(x, 'unknown'), result['labels']))
    result['labels'] = mapped_list
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30002)
