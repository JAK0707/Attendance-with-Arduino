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

# Accessing Cameras
camera_out = cv2.VideoCapture(1)
camera_in = cv2.VideoCapture(2)

# Load Known faces
jai_image = face_recognition.load_image_file("Faces/jai.jpg")
jai_face_encoding = face_recognition.face_encodings(jai_image)[0]

anil_image = face_recognition.load_image_file("Faces/anil.jpg")
anil_face_encoding = face_recognition.face_encodings(anil_image)[0]

# Store all known face encodings and their corresponding names
known_face_encodings = [jai_face_encoding, anil_face_encoding]
known_face_names = ["Jai", "Anil Sir"]

# Initialize variables to store face locations and encodings
face_locations_out = []
face_encodings_out = []
face_locations_in = []
face_encodings_in = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Creating a CSV file for attendance data
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Direction", "Date", "Time"])

# Track detection status for each person
detected_faces_out = {name: False for name in known_face_names}
detected_faces_in = {name: False for name in known_face_names}

while True:
    # Read a frame from the "camera out"
    _, frame_out = camera_out.read()
    # Resize the frame
    small_frame_out = cv2.resize(frame_out, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame_out = cv2.cvtColor(small_frame_out, cv2.COLOR_BGR2RGB)

    # Read a frame from the "camera in"
    _, frame_in = camera_in.read()
    # Resize the frame
    small_frame_in = cv2.resize(frame_in, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame_in = cv2.cvtColor(small_frame_in, cv2.COLOR_BGR2RGB)

    # Recognize faces in "camera out"
    face_locations_out = face_recognition.face_locations(rgb_small_frame_out)
    face_encodings_out = face_recognition.face_encodings(rgb_small_frame_out, face_locations_out)

    # Recognize faces in "camera in"
    face_locations_in = face_recognition.face_locations(rgb_small_frame_in)
    face_encodings_in = face_recognition.face_encodings(rgb_small_frame_in, face_locations_in)

    # Process faces detected in "camera out"
    for face_encoding in face_encodings_out:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame_out, f"{name} Out", bottom_left_corner_of_text, font, fontScale, fontColor, thickness, lineType)

            if not detected_faces_out[name]:
                detected_faces_out[name] = True
                detected_faces_in[name] = False  # Reset in status
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, "Out", current_date, current_time])

    # Process faces detected in "camera in"
    for face_encoding in face_encodings_in:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame_in, f"{name} In", bottom_left_corner_of_text, font, fontScale, fontColor, thickness, lineType)

            if not detected_faces_in[name]:
                detected_faces_in[name] = True
                detected_faces_out[name] = False  # Reset out status
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, "In", current_date, current_time])

    # Check if both Jai and Anil have been detected at least once in "camera out"
    if all(detected_faces_out.values()) and not any(detected_faces_in.values()):
        ledPin.write(1)
    # Check if a face has been detected at least once in "camera in"
    elif any(detected_faces_in.values()):
        ledPin.write(0)

    # Displaying the attendance frame for "camera out"
    cv2.imshow("Attendance Camera Out", frame_out)
    # Displaying the attendance frame for "camera in"
    cv2.imshow("Attendance Camera In", frame_in)

    # Check for 'j' key press to exit the loop and end the program
    if cv2.waitKey(1) & 0xFF == ord('j'):
        break

# Release the cameras and close all OpenCV windows
camera_out.release()
camera_in.release()
cv2.destroyAllWindows()

# Close the CSV file
f.close()
