import pyfirmata2
import time
import csv
import face_recognition
import cv2
import numpy as np
from datetime import datetime

# Initialize the Arduino board
board = pyfirmata2.Arduino('COM4')
ledPin = board.get_pin('d:13:o')

# Accessing Camera
video_capture = cv2.VideoCapture(0)

# Load Known faces
jai_image = face_recognition.load_image_file("Faces/jai.jpg")
jai_face_encoding = face_recognition.face_encodings(jai_image)[0]

anil_image = face_recognition.load_image_file("Faces/anil.jpg")
anil_face_encoding = face_recognition.face_encodings(anil_image)[0]

# Store all known face encodings and their corresponding names
known_face_encodings = [jai_face_encoding, anil_face_encoding]
known_face_names = ["Jai", "Anil Sir"]

# Initialize variables to store face locations and encodings
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Creating a CSV file for attendance data
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Track detection status for each person
detected_faces = {name: False for name in known_face_names}

while True:
    # Read a frame from the camera
    _, frame = video_capture.read()
    # Resize the frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each detected face
    for face_encoding in face_encodings:
        # Compare the face encoding with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            # Get the name of the recognized face
            name = known_face_names[best_match_index]

            # Add a text if a person is present
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottom_left_corner_of_text, font, fontScale, fontColor, thickness,
                        lineType)

            # Update detection status
            if not detected_faces[name]:
                detected_faces[name] = True
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    # Check if both Jai and Anil have been detected at least once
    if all(detected_faces.values()):
        ledPin.write(1)
    else:
        ledPin.write(0)

    # Displaying the attendance frame
    cv2.imshow("Attendance", frame)

    # Check for 'j' key press to exit the loop and end the program
    if cv2.waitKey(1) & 0xFF == ord('j'):
        break

# Release the camera and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Close the CSV file
f.close()
