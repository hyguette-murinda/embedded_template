import cv2
import sqlite3
import numpy as np
from keras.models import load_model

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sunglasses images
sunglasses_male = cv2.imread('sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)
sunglasses_female = cv2.imread('sunglasses-kitty-02.png', cv2.IMREAD_UNCHANGED)

# Load the gender detection model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load the face recognition Haarcascade
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load sign detection model
signModel = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define a scaling factor for the sunglasses size
sunglasses_scale = 0.8  # Adjust as needed

# Gender classification threshold
gender_threshold = 0.6  # Adjust as needed

# Face recognition display settings
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30
nametagHeight = 50
faceRectangleBorderSize = 2
knownTagColor = (100, 180, 0)
unknownTagColor = (0, 0, 255)
knownFaceRectangleColor = knownTagColor
unknownFaceRectangleColor = unknownTagColor

# Initialize face recognition counters
recognition_count = {}
REQUIRED_RECOGNITION_COUNT = 5
face_recognized = False

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while camera.isOpened():
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    if not face_recognized:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each face found
        for (x, y, w, h) in faces:
            # Prepare the input image for gender detection
            face_roi = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Run gender detection
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()

            # Get the predicted gender and confidence
            gender_confidence = genderPreds[0, 0]
            gender = "Male" if gender_confidence > gender_threshold else "Female"

            # Choose the sunglasses based on gender
            if gender == "Male":
                sunglasses = sunglasses_male
            else:
                sunglasses = sunglasses_female

            # Calculate the position and size of the sunglasses
            sunglasses_width = int(sunglasses_scale * w)
            sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])

            # Resize the sunglasses image
            sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

            # Calculate the position to place the sunglasses
            x1 = x + int(w / 2) - int(sunglasses_width / 2)
            x2 = x1 + sunglasses_width
            y1 = y + int(0.55 * h) - sunglasses_height
            y2 = y1 + sunglasses_height

            # Adjust for out-of-bounds positions
            x1 = max(x1, 0)
            x2 = min(x2, frame.shape[1])
            y1 = max(y1, 0)
            y2 = min(y2, frame.shape[0])

            # Create a mask for the sunglasses
            sunglasses_mask = sunglasses_resized[:, :, 3] / 255.0
            frame_roi = frame[y1:y2, x1:x2]

            # Blend the sunglasses with the frame
            for c in range(0, 3):
                frame_roi[:, :, c] = (1.0 - sunglasses_mask) * frame_roi[:, :, c] + sunglasses_mask * sunglasses_resized[:, :, c]

            # Recognize the face
            customer_uid, confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
            customer_name = "Unknown"
            nametagColor = unknownTagColor
            faceRectangleColor = unknownFaceRectangleColor

            # If the face is recognized within the confidence range
            if 60 < confidence < 85:
                try:
                    conn = sqlite3.connect('customer_faces_data.db')
                    c = conn.cursor()
                    c.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_uid,))
                    row = c.fetchone()
                except sqlite3.Error as e:
                    print("SQLite error:", e)
                    row = None
                finally:
                    if conn:
                        conn.close()

                if row:
                    customer_name = row[0].split(" ")[0]
                    nametagColor = knownTagColor
                    faceRectangleColor = knownFaceRectangleColor

                    # Update recognition count
                    if customer_uid not in recognition_count:
                        recognition_count[customer_uid] = 0
                    recognition_count[customer_uid] += 1

                    # Check if the face has been recognized enough times
                    if recognition_count[customer_uid] >= REQUIRED_RECOGNITION_COUNT:
                        face_recognized = True
                        current_customer_uid = customer_uid
                        print(f"Face recognized: {customer_name}")
                        break

            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleColor, faceRectangleBorderSize)
            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, f"{customer_name}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

            # Draw bounding box and label for gender
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"{gender}: {gender_confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if face_recognized:
        # Sign detection logic
        ret, sign_image = camera.read()
        if not ret:
            break

        # Resize and normalize the sign image
        sign_image_resized = cv2.resize(sign_image, (224, 224), interpolation=cv2.INTER_AREA)
        sign_image_array = np.asarray(sign_image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        sign_image_normalized = (sign_image_array / 127.5) - 1

        # Predict the sign
        prediction = signModel.predict(sign_image_normalized)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Display the sign prediction and confidence score
        cv2.putText(sign_image, f"Sign: {class_name[2:]} ({confidence_score*100:.2f}%)", (10, 30), fontFace, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Detection", sign_image)

        # Check if the sign is "OK" and the confidence is above threshold
        if class_name[2:].lower() == "class 1" and confidence_score > 0.75:  # Adjust confidence threshold as needed
            try:
                conn = sqlite3.connect('customer_faces_data.db')
                c = conn.cursor()
                c.execute("UPDATE customers SET confirm = 1 WHERE customer_uid = ?", (current_customer_uid,))
                conn.commit()
                print(f"{customer_name} confirmed")
            except sqlite3.Error as e:
                print("SQLite error:", e)
            finally:
                if conn:
                    conn.close()

            recognition_count = {}  # Reset recognition count
            face_recognized = False  # Resume face detection

    # Display the resulting frame
    if not face_recognized:
        cv2.imshow('Detecting Faces...', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
  