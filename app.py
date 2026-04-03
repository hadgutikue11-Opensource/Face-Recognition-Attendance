import sys  # <--- እዚኣ ኣብታ ቀዳመይቲ መስመር ክትህሉ ኣለዋ
import os
from datetime import datetime, timedelta

# --- 1. Python 3.13 Fix (MUST BE THE VERY FIRST LINES) ---
try:
    import pkg_resources
except ImportError:
    from types import ModuleType
    # Fake pkg_resources setup
    mock_pkg = ModuleType('pkg_resources')
    sys.modules['pkg_resources'] = mock_pkg
    
    def resource_filename(package_or_requirement, resource_name):
        import face_recognition_models
        return os.path.join(face_recognition_models.__path__[0], resource_name)
    
    mock_pkg.resource_filename = resource_filename
# -------------------------------------------------------

import cv2
import face_recognition
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# 2. MongoDB Setup
MONGO_URI = "mongodb+srv://hadgutikue11_db_user:Y8S3FIIzqF1B2kV6@mit-ai-cluster.hsw06yw.mongodb.net/?appName=mit-ai-cluster"
client = MongoClient(MONGO_URI)
db = client['MIT_Attendance']
collection = db['logs']

# 3. Cool-down Tracking Dictionary
last_seen = {}

# 4. Load Known Faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    
    for filename in os.listdir('known_faces'):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join('known_faces', filename)
            img = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
    print(f"✅ Loaded {len(known_face_names)} faces: {known_face_names}")

load_known_faces()

@app.route('/')
def index():
    # ናይ መወዳእታ 10 ዝኣተዉ ተመሃሮ ንምርኣይ
    logs = list(collection.find().sort("time", -1).limit(10))
    formatted_logs = [[l['name'], "", l['time']] for l in logs]
    return render_template('index.html', logs=formatted_logs)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"name": "No Data"})

    try:
        # ምስሊ ካብ Base64 ናብ OpenCV Format ምቕያር
        image_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Face Recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        name = "Unknown"
        now = datetime.now()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
                # --- COOL-DOWN LOGIC (5 Minutes = 300 Seconds) ---
                if name not in last_seen or (now - last_seen[name]).total_seconds() > 300:
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    collection.insert_one({
                        "name": name, 
                        "time": time_str,
                        "status": "Present"
                    })
                    last_seen[name] = now
                    print(f"✅ [DATABASE] Attendance logged for: {name}")
                else:
                    remaining = 300 - (now - last_seen[name]).total_seconds()
                    print(f"⏳ [SKIP] {name} is in cool-down. ({int(remaining)}s left)")
                break

        return jsonify({"name": name})
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"name": "Error"})

if __name__ == '__main__':
    # Ngrok ብትኽክል ንክሰርሕ '127.0.0.1' ንጥቀም
    app.run(host='127.0.0.1', port=5000, debug=True)