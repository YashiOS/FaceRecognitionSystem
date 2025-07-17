"""
Facial Recognition System with User Registration
Compatible with macOS and VS Code
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import json
from datetime import datetime
import sys

class FaceRecognitionSystem:
    def __init__(self, data_dir="face_data"):
        self.data_dir = data_dir
        self.encodings_file = os.path.join(data_dir, "encodings.pkl")
        self.users_file = os.path.join(data_dir, "users.json")
        os.makedirs(data_dir, exist_ok=True)
        
        self.known_encodings = []
        self.known_names = []
        self.users_data = {}
        
        self.load_data()
        print(f"Face Recognition System initialized")
    
    def load_data(self):
        """Load existing face encodings and user data"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                print(f"Loaded {len(self.known_encodings)} existing face encodings")
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users_data = json.load(f)
                print(f"Loaded {len(self.users_data)} user records")
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
    
    def save_data(self):
        """Save face encodings and user data"""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'names': self.known_names
                }, f)
            with open(self.users_file, 'w') as f:
                json.dump(self.users_data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def register_new_user(self, name, frame):
        """Register a new user with their face encoding"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                return False, "❌ No face detected in the image"
            
            if len(face_locations) > 1:
                return False, "❌ Multiple faces detected. Please ensure only one face is visible"
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                return False, "❌ Could not encode the face"
            
            face_encoding = face_encodings[0]
            for i, known_encoding in enumerate(self.known_encodings):
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                if match[0]:
                    existing_user = self.known_names[i]
                    return False, f"❌ This face is already registered as '{existing_user}'"
            
            # Add to known faces
            self.known_encodings.append(face_encoding)
            self.known_names.append(name)
            
            # Save user data
            self.users_data[name] = {
                'registered_date': datetime.now().isoformat(),
                'login_count': 0,
                'last_login': None
            }
            
            # Save data to files
            self.save_data()
            
            return True, f"✅ User '{name}' registered successfully!"
            
        except Exception as e:
            return False, f"❌ Error during registration: {e}"
    
    def recognize_face(self, frame):
        """Recognize faces in the given frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized_faces = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                
                name = "Unknown"
                confidence = 0
                
                if matches and len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        
                        # Update user login data
                        if name in self.users_data:
                            self.users_data[name]['login_count'] += 1
                            self.users_data[name]['last_login'] = datetime.now().isoformat()
                            self.save_data()
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_location
                })
            
            return recognized_faces
            
        except Exception as e:
            print(f"❌ Error during recognition: {e}")
            return []
    
    def draw_face_boxes(self, frame, recognized_faces):
        """Draw bounding boxes and names on the frame"""
        for face in recognized_faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def get_user_info(self, name):
        """Get user information"""
        return self.users_data.get(name, None)
    
    def list_users(self):
        """List all registered users"""
        return list(self.users_data.keys())

def check_camera():
    """Check if camera is available"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible. Please check camera permissions.")
        return False
    cap.release()
    return True

def main():
    print("🎯 Face Recognition System Starting...")
    print("=" * 50)
    
    # Check camera availability
    if not check_camera():
        print("📱 Camera not available. Please ensure:")
        print("1. Camera is connected")
        print("2. VS Code has camera permissions")
        print("3. No other app is using the camera")
        return
    
    # Initialize the face recognition system
    face_system = FaceRecognitionSystem()
    
    # Initialize camera
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
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        if registration_mode:
            # Registration mode
            cv2.putText(frame, f"Registering: {new_user_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture or ESC to cancel", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                success, message = face_system.register_new_user(new_user_name, frame)
                print(message)
                registration_mode = False
                new_user_name = ""
            elif key == 27:  # ESC to cancel
                print("🚫 Registration cancelled")
                registration_mode = False
                new_user_name = ""
        else:
            # Recognition mode
            recognized_faces = face_system.recognize_face(frame)
            frame = face_system.draw_face_boxes(frame, recognized_faces)
            
            # Display instructions
            cv2.putText(frame, "r: Register | u: Users | q: Quit | h: Help", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show recognition results
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
        
        cv2.imshow('Face Recognition System', frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()