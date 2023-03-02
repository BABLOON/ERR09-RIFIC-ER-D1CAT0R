import pandas as pd
import itertools
import cv2

# Read in the data
data = pd.read_csv("optimParameters.csv")

# Define the parameters to optimize over
blur = [0, 3, 5, 7, 9]
alpha = list(range(0,256))
beta = list(range(0,256))
minNeighbors = [1, 2, 3, 4, 5, 6]
scaleFactor = [1.05, 1.1, 1.2]

# Create a list of all parameter combinations
params = list(itertools.product(blur, alpha, beta, minNeighbors, scaleFactor))

# Initialize variables to keep track of the best parameters and highest confidence score
best_params = None
best_score = 0

# Loop through each parameter combination
for param in params:
    # Unpack the parameters
    blur_val, alpha_val, beta_val, minNeighbors_val, scaleFactor_val = param
    
    # Run the object detector with the current parameters
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    face_cascade.load('haarcascade_frontalface_alt2.xml')
    detected_faces = []
    for index, row in data.iterrows():
        img = cv2.imread(row['filename'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayBlur = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
        grayNorm = cv2.normalize(grayBlur, None, alpha=alpha_val, beta=beta_val, norm_type=cv2.NORM_MINMAX)
        faces = face_cascade.detectMultiScale(grayNorm, scaleFactor=scaleFactor_val, minNeighbors=minNeighbors_val)
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            detected_faces.append(face)
    
    # Calculate the confidence score for the current parameters
    confidence = sum([row['confidence'] for index, row in data.iterrows() if row['filename'] in detected_faces])
    
    # Update the best parameters and highest confidence score if the current score is better
    if confidence > best_score:
        best_params = param
        best_score = confidence

# Print the best parameters and highest confidence score
print(f"Best parameters: blur={best_params[0]}, alpha={best_params[1]}, beta={best_params[2]}, minNeighbors={best_params[3]}, scaleFactor={best_params[4]}")
print(f"Highest confidence score: {best_score}")
