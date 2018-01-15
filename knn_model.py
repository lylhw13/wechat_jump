#   -*- coding: utf-8 -*-

import cv2
import numpy as np
img = cv2.imread('data/numbers_1440_2560.jpg')
img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(img_cvt, (5, 5), 0)

ratio = 15

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
_,cons, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

rects = np.array([cv2.boundingRect(con) for con in cons ])
w_max = max(rects[:,2])
h_max = max(rects[:,3])

response = []
samples = np.empty((0, ratio*ratio))

for rect in rects:
    if rect[2] > 0.95 * w_max or rect[3] > 0.95 * h_max:
        x = rect[0]
        y = rect[1]
        cv2.rectangle(img,(x,y),(x + w_max, y + h_max),(0,0,255),2)
        roi = thresh[y:y+h_max,x:x+w_max]
        roismall = cv2.resize(roi,(ratio, ratio))
        cv2.imshow('norm',img)
        key = cv2.waitKey(0)
        if key==27:#'Esc'
            break
        response.append(int(chr(key)))
        print('number is {0}, x= {1}, y= {2}'.format(int(chr(key)), x, y))
        sample = roismall.reshape((1,ratio*ratio))
        samples = np.append(samples,sample,0)

responses = np.array(response, dtype=np.float32)
responses = responses.reshape((responses.size,1))

np.savetxt('data/X_train_1440_2560.txt',samples)
np.savetxt('data/y_train_1440_2560.txt',response)

cv2.destroyAllWindows()