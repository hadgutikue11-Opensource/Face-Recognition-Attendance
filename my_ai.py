import face_recognition
import cv2
import numpy as np
import os

def load_known_faces(directory):
    known_encodings = []
    known_names = []
    # Automatically learn every photo in the folder (HTG, Gerush, etc.)
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if len(encoding) > 0:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def get_face_data(frame, known_encodings, known_names):
    # Resize for speed (0.25) as per your script
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown Person"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale back up (x4) to match original frame size
        results.append(((top*4, right*4, bottom*4, left*4), name))
    
    return results