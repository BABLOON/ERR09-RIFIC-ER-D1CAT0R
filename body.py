import cv2

# load the upper body classifier
upper_body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


# Open a video stream
cap = cv2.VideoCapture(0)

while True:

    # Read a frame from the video stream
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Normalize the pixel values in the grayscale image
    norm_gray = cv2.normalize(blur_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    
    # detect upper bodies in the image
    upper_bodies = upper_body_classifier.detectMultiScale(blur_gray, scaleFactor=1.01, minNeighbors=7, minSize=(90, 90), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw a rectangle around each detected body
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with the detected bodies
    cv2.imshow('frame', frame)
    
    # Exit the program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()