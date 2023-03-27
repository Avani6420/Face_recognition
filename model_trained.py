import cv2 as cv
import numpy as np
import os

# get the all folder name
people_list = ['Elon musk', 'Bill gates', 'Anita borg']
DIR = 'G:\\Temp_python\\Face_recognition\\data_set'

features = [] # store corresponding imgs
lables = [] # correspondig lable for img

# face detector classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# set to the training set
def create_tain():
    # each folder and grabbing the path to the folder
    for person in people_list:
        path =  os.path.join(DIR, person) # prepare img path
        label = people_list.index(person)

        # loop for a every img in folder
        for img in os.listdir(path): # get the img path
            img_path = os.path.join(path, img)

            # read the img from the img_path
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #img conver in gray scale img

            # detect the face and find the face on the img
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # loop over the every face detact
            for (x,y, h, w) in face_rect:
                face_roi = gray[y:y+h, x:x+w] # region of intrest
                # append img and labels
                features.append(face_roi)
                lables.append(label)

create_tain()

# list convert to numpy array
features = np.array(features, dtype='object')
lables = np.array(lables)

# face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recogizer on the featureslist and the labels list
face_recognizer.train(features, lables)
# save the train model files
# face_recognizer.save('model_trained.yml')
# np.save('features.npy', features)
# np.save('lables.npy', lables)