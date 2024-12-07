import tensorflow as tf
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import dlib
import cv2
import time
from matplotlib import pyplot as plt
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def endprogram():
    print("\nProgram terminated!")
    sys.exit()


def fulltraining():
    import Model as mm


def testing():
    global testing_screen
    testing_screen = Toplevel(main_screen)
    testing_screen.title("Testing")
    # login_screen.geometry("400x300")
    testing_screen.geometry("600x450+650+150")
    testing_screen.minsize(120, 1)
    testing_screen.maxsize(1604, 881)
    testing_screen.resizable(1, 1)
    testing_screen.configure(bg='cyan')
    # login_screen.title("New Toplevel")

    Label(testing_screen, text='''Upload Video''', disabledforeground="#a3a3a3",
          foreground="#000000", width="300", height="2", bg='cyan', font=("Calibri", 16)).pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Button(testing_screen, text='''Upload Video''', font=(
        'Verdana', 15), height="2", width="30", bg='cyan', command=imgtest).pack()


global affect


def imgtest():
    import_file_path = filedialog.askopenfilename()

    model = load_model('Model/deepfake-detection-model.h5')

    input_shape = (128, 128, 3)
    pr_data = []
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(import_file_path)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate) + 1) * 1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                data = data.reshape(-1, 128, 128, 3)
                # print(model.predict_classes(data))
                result = model.predict(data)
                ind = np.argmax(result)
                out=''

                if ind == 0:

                    out = "Fake"

                elif ind == 1:

                    out = "Real"

                print(out)

                cv2.putText(crop_img, out, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.imshow("cropped", cropped)
                cv2.imshow("Output", crop_img)
                cv2.waitKey(0)


def image():
    import_file_path = filedialog.askopenfilename()

    model = load_model('Model/deepfake-detection-model.h5')


    #cap = cv2.VideoCapture(import_file_path)
    #frameRate = cap.get(5)
    crop_img = cv2.imread(import_file_path)
    data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
    data = data.reshape(-1, 128, 128, 3)
    # print(model.predict_classes(data))
    result = model.predict(data)
    ind = np.argmax(result)
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





def result():
    import warnings
    warnings.filterwarnings('ignore')




def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure(bg='cyan')
    main_screen.title("DeepFake  Detection  ")

    Label(text="DeepFake Detection", width="300", height="5", bg='cyan', font=("Calibri", 16)).pack()

    Button(text="Training", font=(
        'Verdana', 15), height="2", width="30", command=fulltraining, highlightcolor="black", bg='cyan').pack(side=TOP)

    Label(text="").pack()
    Button(text="Image", font=(
        'Verdana', 15), height="2", width="30", bg='cyan', command=image).pack(side=TOP)

    Label(text="").pack()
    Button(text="Video", font=(
        'Verdana', 15), height="2", width="30", bg='cyan', command=testing).pack(side=TOP)

    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()
