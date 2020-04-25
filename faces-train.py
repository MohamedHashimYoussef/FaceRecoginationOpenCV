import os
import cv2
from PIL import Image
import numpy as np
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR , "images")
face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
labels_id = {}
x_train = []
y_labels = []
for root , dirs , files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root , file)
            label = os.path.basename(root).replace(" " , "-").lower()
            if not label in labels_id:
                labels_id[label] = current_id
                current_id += 1
            id_ = labels_id[label]
            pil_image = Image.open(path).convert("L") # read image and convert it to gray
            size = (550 , 550)
            final_img = pil_image.resize(size , Image.ANTIALIAS)
            image_arr = np.array(final_img, "uint8")
            faces = face_cascade.detectMultiScale(image_arr , scaleFactor=1.5 , minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_arr[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(labels_id)
# print(y_labels)
# print(len(x_train))

with open("labels.pickle" , "wb") as f:
    pickle.dump(labels_id , f)

recognizer.train(x_train , np.array(y_labels))
recognizer.write("trainer.yml")

