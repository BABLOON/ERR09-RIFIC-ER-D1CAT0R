import pandas as pd
import random
import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

#training image path
image_folder = 'C:/Users/Rowan/Documents/projects/ERR09-RIFIC-ER-D1CAT0R/training_images/lfw-deepfunneled'

#create dataset
dataset = pd.DataFrame(columns=['confidence', 'blur', 'alpha', 'beta', 'minNeighbors', 'scaleFactor'])

# Iterate through all subdirectories in the larger folder
for folder_name in os.listdir(image_folder):
    folder_path = os.path.join(image_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Loop through each image file in the directory
    for filename in os.listdir(folder_path):
        if not filename.endswith('.jpg'):
            continue

        # Load the image
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        #generate values
        blur = (random.randint(1, 5) * 2) - 1
        alpha = random.randint(0,255)
        beta = random.randint(0,255)
        minNeighbors = random.randint(0,10)
        scaleFactor = random.uniform(1.03, 1.1)
        confidence = 0

        # Image processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayBlur = cv2.GaussianBlur(gray, (blur, blur), 0)
        grayNormalized = cv2.normalize(grayBlur, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)

        faces = face_cascade.detectMultiScale3(grayNormalized, scaleFactor=scaleFactor, minNeighbors=minNeighbors, outputRejectLevels=True)
        eyes = eyes_cascade.detectMultiScale3(grayNormalized, scaleFactor=scaleFactor, minNeighbors=minNeighbors, outputRejectLevels=True)

        if len(eyes[0]) == 0:
            eyes_confidence = 0
        else:
            eyes_confidence = np.max(eyes[2])

        if len(faces[0]) == 0:
            faces_confidence = 0
        else:
            faces_confidence = np.max(faces[2])

        confidence = eyes_confidence + faces_confidence

        dataset = dataset.append({'confidence': confidence, 'blur': blur, 'alpha': alpha, 'beta': beta, 'minNeighbors': minNeighbors, 'scaleFactor': scaleFactor}, ignore_index=True)




#return dataset file
dataset.to_csv('optimParameters.csv', index=False)