# face_recognition_checkin.py

import face_recognition
import pickle
import cv2
import numpy as np
import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Load known faces and embeddings
ENCODINGS_FILE = 'encodings.pickle'
data = pickle.loads(open(ENCODINGS_FILE, "rb").read())

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Google Sheets setup
SERVICE_ACCOUNT_FILE = 'credentials.json'  # Path to your service account credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = 'YOUR_SPREADSHEET_ID'  # Replace with your spreadsheet ID

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('sheets', 'v4', credentials=credentials)
sheet = service.spreadsheets()

# Create a set to keep track of check-ins for the day
checked_in_today = set()
current_date = datetime.datetime.now().date()

def has_checked_in_today(name):
    """Check if the person has already checked in today."""
    return name in checked_in_today

def record_check_in(name):
    """Record the check-in in the Google Spreadsheet."""
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    # Append the data to the spreadsheet
    values = [[date_str, time_str, name]]
    body = {'values': values}
    result = sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range='Sheet1!A:C',
        valueInputOption='RAW',
        body=body
    ).execute()
    print(f"Recorded check-in for {name} at {time_str}")
    checked_in_today.add(name)

while True:
    # Reset check-ins at midnight
    if datetime.datetime.now().date() != current_date:
        checked_in_today.clear()
        current_date = datetime.datetime.now().date()

    ret, frame = video_capture.read()
    if not ret:
        break
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert from BGR to RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encodings with known faces
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
        name = "Unknown"

        # Use the known face with the smallest distance if any
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data["names"][best_match_index]
            if not has_checked_in_today(name):
                record_check_in(name)
        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
