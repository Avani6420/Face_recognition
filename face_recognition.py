import cv2 as cv
import numpy as np

def resize_frame(frame, scale=0.50):
    # re_scale for the exiceting video , image, live video
    width = int(frame.shape[1] * scale) # set the floating point value of the screen
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# face detector classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')
# mapping list names
people_list = ['Anita borg', 'Bill gates', 'Elon musk']

# load features and labels array
# features = np.load('features.npy') # if load then (, allow_pickle=True)
# lables = np.load('lables.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('model_trained.yml')

# get the img
img = cv.imread(r'G:\\Temp_python\\Face_recognition\\val\\Bill gates\\bill-gates.jpeg')

frame = resize_frame(img)
# BGR TO GRAY
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detact the face img
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, h, w) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    # usig face recogniztion get label and confidence value
    label, confidence = face_recognizer.predict(faces_roi)
    print("label: ", people_list[label], " confidence: ", confidence)

    # put text on the img
    cv.putText(img, str(people_list[label]), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness= 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness= 2)

cv.imshow('Detected_img', img)
cv.waitKey(0)
