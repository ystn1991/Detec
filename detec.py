import os
import cv2
import numpy as np

# Path to the base folder containing member folders
base_folder = './group_members/'

# รายชื่อสมาชิกและตำแหน่ง
members = [
    ('Muay', 'P1_CEO'),
    ('Fahfon', 'P2_UXUI'),
    ('Nine', 'P3_Front1'),
    ('Mhooyong', 'P4_Front2'),
    ('First', 'P5_back1'),
    ('Jay', 'P6_back2'),
    ('Golf', 'P7_se'),
    ('Boss', 'P8_sa')
]

# Prepare training data
def prepare_training_data(base_folder):
    images = []
    labels = []
    label_dict = {}

    # Loop through each member folder
    for label, (member_name, position) in enumerate(members):
        member_folder = os.path.join(base_folder, position)  # Use position as folder name

        # Ensure it's a directory
        if os.path.isdir(member_folder):
            label_dict[label] = (position, member_name)  # Map label to (position, member name)
            for image_name in os.listdir(member_folder):
                image_path = os.path.join(member_folder, image_name)
                image = cv2.imread(image_path)

                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                images.append(gray_image)
                labels.append(label)

    return images, labels, label_dict

# Load training data
images, labels, label_dict = prepare_training_data(base_folder)

# Create and train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, np.array(labels))

# Start video capture for recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # No faces detected
        cv2.putText(frame, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            label, confidence = model.predict(roi_gray)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the label in the desired format
            if confidence < 100:
                position, member_name = label_dict[label]
                formatted_label = f"{position}_{member_name}"
            else:
                formatted_label = "Unknown"

            cv2.putText(frame, formatted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame with detections
    cv2.imshow('Video Stream - Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
