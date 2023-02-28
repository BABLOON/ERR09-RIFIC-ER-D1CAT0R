import math
import cv2

# Load the Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

neighborThresh = 3

# Camera parameters
FOV = 60.0 # Camera field of view in degrees
WIDTH = 640 # Camera resolution width in pixels
HEIGHT = 480 # Camera resolution height in pixels
FOCAL_LENGTH = WIDTH / (2 * math.tan(math.radians(FOV / 2))) # Camera lens focal length

# Face size in meters
FACE_WIDTH = 0.15 # Typical width of a face in meters
FACE_HEIGHT = 0.2 # Typical height of a face in meters


while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=neighborThresh)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        # Calculate the distance from the camera to the face
        pixel_width = w
        distance = (FACE_WIDTH * FOCAL_LENGTH) / pixel_width
        
        # Draw the face box on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the distance from the face under the face box
        cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        # Display current min_neighbors value
        cv2.putText(frame, f"Min neighbor threshold: {neighborThresh}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('[') and neighborThresh > 1:
        neighborThresh -= 1
    elif key == ord(']'):
        neighborThresh += 1
# Release the VideoCapture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
