import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model('Model/deepfake-detection-model.h5')

input_shape = (128, 128, 3)
pr_data = []
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture('train_sample_videos/abarnvbtwb.mp4')
frameRate = cap.get(5)
while cap.isOpened():
    frameId = cap.get(1)
    ret, frame = cap.read()
    if ret != True:
        break
    if frameId % ((int(frameRate)+1)*1) == 0:
        face_rects, scores, idx = detector.run(frame, 0)
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            crop_img = frame[y1:y2, x1:x2]
            data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
            data = data.reshape(-1, 128, 128, 3)
            #print(model.predict_classes(data))
            result = model.predict(data)
            print(result)
            out=''

            ind = np.argmax(result)
            print(ind)
            out = ''

            if ind == 0:

                out = "Fake"

            elif ind == 1:

                out = "Real"

            print(out)

            cv2.putText(crop_img, out, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.imshow("cropped", cropped)
            cv2.imshow("Output", crop_img)
            cv2.waitKey(0)

