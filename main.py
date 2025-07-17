from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from face_system import FaceRecognitionSystem

app = FastAPI()
face_system = FaceRecognitionSystem()

from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "FastAPI app on Render works!"}


@app.post("/register")
async def register_user(name: str = Form(...), image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        success, message = face_system.register_new_user(name, frame)
        return {"success": success, "message": message}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = face_system.recognize_face(frame)
        return {"success": True, "recognized_faces": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/user-list")
def list_users():
    users = face_system.list_users()
    return {"users": users}

@app.get("/user-info/{name}")
def get_user_info(name: str):
    user = face_system.get_user_info(name)
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found"})
    return user

@app.get("/health")
def health_check():
    return {"status": "ok"}
