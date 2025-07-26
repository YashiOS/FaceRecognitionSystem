from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

from face_system import FaceRecognitionSystem

app = FastAPI()

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("fastapifacial-firebase-adminsdk-fbsvc-e2447ecd93.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load face system
face_system = FaceRecognitionSystem()


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
        filtered_results = [face for face in results if face['name'] != "Unknown"]

        timestamp = datetime.utcnow().isoformat()
        for face in filtered_results:
            db.collection("recognition_logs").add({
                "name": face['name'],
                "confidence": round(face['confidence'], 4),
                "recognized_at": timestamp
            })

        return {
            "success": True,
            "recognized_faces": results  # Contains name, confidence, and location
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/login")
async def login_user(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = face_system.recognize_face(frame)
        known_faces = [face for face in results if face['name'] != "Unknown"]

        timestamp = datetime.utcnow().isoformat()

        if known_faces:
            for face in known_faces:
                db.collection("login_logs").add({
                    "name": face['name'],
                    "confidence": round(face['confidence'], 4),
                    "login_at": timestamp
                })

            return {
                "success": True,
                "message": "Login successful",
                "users": known_faces  # Includes confidence
            }
        else:
            return {
                "success": False,
                "message": "No known faces found. Login failed.",
                "users": results  # Still returns attempted faces with confidence
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/user-list")
def list_users():
    users_ref = db.collection("users").stream()
    users = [user.id for user in users_ref]
    return {"users": users}


@app.get("/user-info/{name}")
def get_user_info(name: str):
    doc = db.collection("users").document(name).get()
    if doc.exists:
        return doc.to_dict()
    return JSONResponse(status_code=404, content={"error": "User not found"})


@app.get("/health")
def health_check():
    return {"status": "ok"}
