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

@app.post("/detect")
async def detect_human(file_path):
    result = human_detection(file_path)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30002)
