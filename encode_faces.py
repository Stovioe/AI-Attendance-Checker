# encode_faces.py

import face_recognition
import pickle
import cv2
import os

KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'encodings.pickle'

known_encodings = []
known_names = []

# Loop over the known faces and encode them
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            encoding = encodings[0]
            known_encodings.append(encoding)
            known_names.append(name)
        else:
            print(f"No face found in {filepath}")

# Save the encodings and names to a file
data = {"encodings": known_encodings, "names": known_names}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("Encoding complete")
