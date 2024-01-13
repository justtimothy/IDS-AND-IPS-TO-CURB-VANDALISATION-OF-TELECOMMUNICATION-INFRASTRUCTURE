import cv2
import os
import time
import datetime
import numpy as np

from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
from roboflow import Roboflow

# initialize Roboflow
rf = Roboflow(api_key="s68JQSR4jNTVJBkyedW")
project = rf.workspace().project("telecommunication")
model = project.version(1).model

# Load the saved faces and their corresponding labels from the file path
faces_path = os.path.join('C:\\Users\\LENOVO\\Desktop', 'Face-Recognition-Based-Attendance-System-main', 'data')
faces = []
labels = []
for i, filename in enumerate(sorted(os.listdir(faces_path))):
    face = cv2.imread(os.path.join(faces_path, filename))
    face = cv2.resize(face, (160, 160))
    face = face.flatten() / 255.0
    faces.append(face)
    labels.append(i)
faces = np.array(faces)
labels = np.array(labels)

# Load the known faces
known_faces = []
known_names = []
for name in os.listdir(known_face_path):
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        face = cv2.imread(os.path.join(known_face_path, name))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (92, 112))
        known_faces.append(face)
        known_names.append(name[:-4])

# Initialize the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer
face_recognizer.train(known_faces, np.array(known_names))

# Load the FaceNet model
facenet_model = load_model('facenet_keras.h5')

# Define the threshold for face recognition
threshold = 1.2


# get the current date
now = datetime.datetime.now()
date_folder = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.strftime("%S")}'

# initialize the camera
cap = cv2.VideoCapture(0)


# define functions for each class name
def ANIMAL():
    print("ANIMAL detected")

def BREAKING():
    print("BREAKING detected")

# add more functions for each class name here...

# dictionary to map class names to functions
class_to_func_extras_object = {
    "ANIMAL": ANIMAL,
    "Building": lambda: print("Building"),
    "HOUSE": lambda: print("HOUSE"),
    "House": lambda: print("HOUSE"),
    "Man": lambda: print("Man"),
    "Vehicle": lambda: print("Vehicle"),
    "Tree": lambda: print("Tree")
    }

class_to_func_extras = {
    "CUTLASS": lambda: print("CUTLASS"),
    "FIRE": lambda: print("FIRE"),
    "FOG": lambda: print("FOG"),
    "KNIFE(CLEAVER)": lambda: print("KNIFE(CLEAVER)"),
    "KNIFE(SCABBARD)": lambda: print("KNIFE(SCABBARD)"),
    "Land vehicle": lambda: print("Land vehicle"),
    "SPEAR": lambda: print("SPEAR"),
    "SICKLE": lambda: print("SICKLE"),
    "SMOKE": lambda: print("SMOKE"),
    "WEAPON(GUN)": lambda: print("WEAPON(GUN)"),
    "WEAPON(KATANA)": lambda: print("WEAPON(KATANA)"),
    "WEAPON(KNIFE)": lambda: print("WEAPON(KNIFE)"),
    "bomb": lambda: print("bomb"),
    "fire": lambda: print("fire"),
    }

class_to_func_human = {
    "PERSON": lambda: print("PERSON"),
    "Person": lambda: print("Person"),
}

class_to_func_telecommunication_objects = {
    "GENERATOR": lambda: print("GENERATOR"),
    "GSM ANTENNA": lambda: print("GSM ANTENNA"),
    "MAST": lambda: print("MAST"),
    "MICROWAVE ANTENNA": lambda: print("MICROWAVE ANTENNA"),
    "MIKANO GENERATOR": lambda: print("MIKANO GENERATOR"),
    }

class_to_func_actions = {
    "BREAKING":BREAKING,
    "DESTROYING": lambda: print("DESTROYING"),
    "FALL": lambda: print("FALL"),
    "THROWING": lambda: print("THROWING"),
    }

class_to_func_extras_class_labels = [] # create an empty list to store class labels
class_to_func_actions_class_labels = [] # create an empty list to store class labels

class_to_func_human_class_labels = set() # create an empty set to store class labels
class_to_func_telecommunication_objects_class_labels = set() # create an empty set to store class labels


# Use the first camera
for i in range(10):  # Capture 10 frames from the first
    ret, frame = cap.read()



    # Perform object detection on the frame
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # Apply non-max suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes on the frame
    name =0

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the class name on the frame
        label = str(classes[class_ids[i]])
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if a prediction has a class name of elements in class_to_func_telecommunication_objects
        if label in class_to_func_telecommunication_objects:       
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            objectDetail = box.append[name] 
            name += 1

            # Adds element without repetition to set class_to_func_telecommunication_objects_class_labels
            class_to_func_telecommunication_objects_class_labels.add(objectDetail)

            # Make console message
            class_to_func_extras[label]()

            # Create or append to folder with timestamp as name
            folder_name = f"{date_folder}/{class_to_func_telecommunication_objects_class_labels}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Save object to folder
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{folder_name}/frame_{timestamp}.jpg'
            cv2.imwrite(filename, frame[start_y:end_y, start_x:end_x])


    # Display the frame
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()

# close all windows
cv2.destroyAllWindows()


# Use the second camera

# initialize the camera
cap = cv2.VideoCapture(1)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # make a prediction
    result = model.predict(frame, confidence=40, overlap=30)
    predictions = result.predictions

    # draw the bounding boxes on the frame
    for prediction in predictions:
        cv2.rectangle(frame, (prediction['xmin'], prediction['ymin']), (prediction['xmax'], prediction['ymax']), (0,255,0), 2)


        # call the appropriate function based on the class name extras
        if prediction['class_name'] in class_to_func_extras:
            class_to_func_extras[prediction['class_name']]()
            
        # call the appropriate function based on the class_to_func_extras_object
        if prediction['class_name'] in class_to_func_extras_object:
            class_to_func_extras_object[prediction['class_name']]()
            print(' detected')
            
        # call the appropriate function based on the class name class_to_func_telecommunication_objects
        if prediction['class_name'] in class_to_func_telecommunication_objects:
            class_to_func_telecommunication_objects[prediction['class_name']]()

        # call the appropriate function based on the class human
        if prediction['class_name'] in class_to_func_human:
            class_to_func_human[prediction['class_name']]()

            # check authorization            
            # Detect faces in the camera input
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_classifier.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Loop through the detected faces
            for (x, y, w, h) in faces:
                # Recognize the face
                face = cv2.resize(gray_frame[y:y+h, x:x+w], (92, 112))
                label, confidence = face_recognizer.predict(face)


                # Check if the face is recognized
                if label in known_names:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Display the name of the recognized person
                    cv2.putText(frame, known_names[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                else:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    # Display "Unknown" for unrecognized faces
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Create or append to folder with timestamp as name
                    folder_name = f"{date_folder}/{unauthorized}"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)

                    # Save frame to folder
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'{folder_name}/frame_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)

                    # Save message to file
                    with open(f'{folder_name}/message.txt', 'a') as f:                            
                        f.write(f"{prediction['class_name']} was an unauthorized person at {timestamp}\n")
                            

        # call the appropriate function based on the class human
        if prediction['class_name'] in class_to_func_actions:
            class_to_func_actions_class_labels[prediction['class_name']]()

            # Create or append to folder with timestamp as name
            folder_name = f"{date_folder}/{vandal}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Save frame to folder
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{folder_name}/frame_{timestamp}.jpg'
            cv2.imwrite(filename, frame)

            # Save message to file
            with open(f'{folder_name}/message.txt', 'a') as f:                            
                f.write(f"{prediction['class_name']} occured at {timestamp}\n")
    

    # display the resulting frame
    cv2.imshow('frame', frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release the camera
cap.release()

# close all windows
cv2.destroyAllWindows()