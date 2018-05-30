# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import os
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

base_path = "c:/base"
def face_detection(path, filename):
    img = cv.imread(path  + "/" + filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height, width, channels = img.shape

    # 세로사진만 취급한다.
    if height < width:
        return

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 한명이 나온 사진만.
    if len(faces) != 1:
        return

    x, y, w, h = faces[0]

    # 얼굴이 전체의 1/10 이하여야 전신이 나올 수 있다.
    if height / h > 10:
        return

    # 얼굴이 위쪽에 치우쳐 있어야 전신이 나올 수 있다.
    if x + h > height / 3:
        return

    cv.imshow('img', img)
    cv.waitKey(0)

def main():
    begin = False
    for (path, dir, files) in os.walk(base_path):
        for filename in files:
            if not begin and filename == '':
                begin = True

            if begin:
                if filename.endswith(".jpg"):
                    try :
                        print(filename)
                        face_detection(path, filename)
                    except:
                        pass

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
