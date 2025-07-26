import cv2
import face_recognition
import numpy as np
import os
import pickle
import json
from datetime import datetime
import requests

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("fastapifacial-firebase-adminsdk-fbsvc-e2447ecd93.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


class FaceRecognitionSystem:
    def __init__(self, data_dir="face_data"):
        self.data_dir = data_dir
        self.encodings_file = os.path.join(data_dir, "encodings.pkl")
        self.users_file = os.path.join(data_dir, "users.json")
        os.makedirs(data_dir, exist_ok=True)

        self.known_encodings_dict = {}  # {name: [encoding1, encoding2, ...]}
        self.users_data = {}

        self.load_data()
        print("✅ Face Recognition System initialized")

    def load_data(self):
        """Load encodings and user info, validate shape"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    raw_data = pickle.load(f)

                for name, enc_list in raw_data.items():
                    valid_encodings = []
                    for enc in enc_list:
                        enc_np = np.array(enc, dtype=np.float32)
                        if enc_np.shape == (128,):
                            valid_encodings.append(enc_np)
                        else:
                            print(f"⚠️ Skipping invalid encoding for {name}")
                    self.known_encodings_dict[name] = valid_encodings

                print(f"Loaded encodings for {len(self.known_encodings_dict)} users")

            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users_data = json.load(f)
                print(f"Loaded {len(self.users_data)} user records")

        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def save_data(self):
        """Save encodings and user info"""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.known_encodings_dict, f)
            with open(self.users_file, 'w') as f:
                json.dump(self.users_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def save_user_to_firestore(self, name):
        """Sync user to Firestore"""
        try:
            user_info = self.users_data.get(name)
            if user_info:
                db.collection("users").document(name).set({
                    "name": name,
                    "registered_date": user_info['registered_date'],
                    "login_count": user_info['login_count'],
                    "last_login": user_info['last_login']
                })
                print(f"☁️ Synced user '{name}' to Firestore")
        except Exception as e:
            print(f"❌ Firestore sync failed for '{name}': {e}")

    def register_new_user(self, name, frame):
        """Register a new user with a face frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if not face_locations:
                return False, "❌ No face detected in image"
            if len(face_locations) > 1:
                return False, "❌ Multiple faces detected"

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                return False, "❌ Failed to encode face"

            face_encoding = np.array(face_encodings[0], dtype=np.float32)

            for user, enc_list in self.known_encodings_dict.items():
                for known_encoding in enc_list:
                    match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.45)
                    if match[0]:
                        return False, f"❌ Face already registered as '{user}'"

            if name not in self.known_encodings_dict:
                self.known_encodings_dict[name] = []
            self.known_encodings_dict[name].append(face_encoding)

            if len(self.known_encodings_dict[name]) > 5:
                self.known_encodings_dict[name] = self.known_encodings_dict[name][-5:]

            self.users_data[name] = {
                "registered_date": datetime.now().isoformat(),
                "login_count": 0,
                "last_login": None
            }

            self.save_data()
            self.save_user_to_firestore(name)

            return True, f"✅ User '{name}' registered successfully!"
        except Exception as e:
            return False, f"❌ Registration error: {e}"

    def recognize_face(self, frame):
        """Recognize faces in frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_faces = []

            for face_encoding, face_location in zip(face_encodings, face_locations):
                all_known_encodings = []
                all_known_names = []

                for name, enc_list in self.known_encodings_dict.items():
                    all_known_encodings.extend(enc_list)
                    all_known_names.extend([name] * len(enc_list))

                if not all_known_encodings:
                    recognized_faces.append({
                        "name": "Unknown",
                        "confidence": 0,
                        "location": face_location
                    })
                    continue

                matches = face_recognition.compare_faces(all_known_encodings, face_encoding, tolerance=0.45)
                face_distances = face_recognition.face_distance(all_known_encodings, face_encoding)

                name = "Unknown"
                confidence = 0

                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        confidence = 1 - face_distances[best_match_index]
                        if confidence >= 0.65:
                            name = all_known_names[best_match_index]

                            if name in self.users_data:
                                self.users_data[name]['login_count'] += 1
                                self.users_data[name]['last_login'] = datetime.now().isoformat()
                                self.save_data()

                recognized_faces.append({
                    'name': name,
                    'confidence': round(confidence, 2),
                    'location': face_location
                })

            return recognized_faces
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return []

    def draw_face_boxes(self, frame, recognized_faces):
        """Draw boxes and labels on frame"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def get_user_info(self, name):
        return self.users_data.get(name)

    def list_users(self):
        return list(self.users_data.keys())



def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible. Please check camera permissions.")
        return False
    cap.release()
    return True

def login_via_api(frame):
    """Send current frame to /login API and display results"""
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print("❌ Failed to encode frame for login")
        return

    files = {'image': ('login.jpg', buffer.tobytes(), 'image/jpeg')}

    try:
        response = requests.post("http://69.62.78.177:8000/login", files=files)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Login successful:")
                for user in data['users']:
                    confidence_pct = user['confidence'] * 100
                    status = "🟢" if confidence_pct >= 80 else "🟡" if confidence_pct >= 65 else "🔴"
                    print(f"  {status} {user['name']} (confidence: {confidence_pct:.1f}%)")
            else:
                print("❌ Login failed:", data['message'])
        else:
            print(f"❌ Login API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to call login API: {e}")


def main():
    print("🎯 Face Recognition System Starting...")
    print("=" * 50)

    if not check_camera():
        print("📱 Camera not available. Please ensure:")
        print("1. Camera is connected")
        print("2. VS Code has camera permissions")
        print("3. No other app is using the camera")
        return

    face_system = FaceRecognitionSystem()
    cap = cv2.VideoCapture(0)

    print("\n🎥 Camera initialized successfully!")
    print("\n📋 COMMANDS:")
    print("- Press 'r' to register a new user")
    print("- Press 'u' to show user list")
    print("- Press 'q' to quit")
    print("- Press 'h' for help")
    print("=" * 50)

    registration_mode = False
    new_user_name = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from camera")
            break

        frame = cv2.flip(frame, 1)

        if registration_mode:
            cv2.putText(frame, f"Registering: {new_user_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture or ESC to cancel", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                success, message = face_system.register_new_user(new_user_name, frame)
                print(message)
                registration_mode = False
                new_user_name = ""
            elif key == 27:
                print("🚫 Registration cancelled")
                registration_mode = False
                new_user_name = ""
        else:
            recognized_faces = face_system.recognize_face(frame)
            frame = face_system.draw_face_boxes(frame, recognized_faces)
            cv2.putText(frame, "r: Register | u: Users | l: Login | q: Quit | h: Help",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for i, face in enumerate(recognized_faces):
                if face['name'] != "Unknown":
                    cv2.putText(frame, f"Welcome back, {face['name']}!",
                               (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                new_user_name = input("\n👤 Enter name for new user: ").strip()
                if new_user_name:
                    registration_mode = True
                    print(f"📸 Registration mode activated for: {new_user_name}")
                    print("Position your face in the camera and press SPACE to capture")
                else:
                    print("❌ Invalid name entered")
            elif key == ord('u'):
                users = face_system.list_users()
                print(f"\n👥 REGISTERED USERS ({len(users)}):")
                print("-" * 40)
                for user in users:
                    info = face_system.get_user_info(user)
                    last_login = info['last_login']
                    if last_login:
                        last_login = datetime.fromisoformat(last_login).strftime("%Y-%m-%d %H:%M")
                    else:
                        last_login = "Never"
                    print(f"• {user}")
                    print(f"  Logins: {info['login_count']}")
                    print(f"  Last login: {last_login}")
                    print()
            elif key == ord('h'):
                print("\n📋 HELP:")
                print("- r: Register new user")
                print("- u: Show all registered users")
                print("- q: Quit application")
                print("- h: Show this help")
                print("- During registration: SPACE to capture, ESC to cancel")
            elif key == ord('q'):
                print("\n👋 Goodbye!")
                break
            elif key == ord('l'):
                print("\n🔐 Attempting login via API...")
                login_via_api(frame)
        cv2.imshow('Face Recognition System', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
