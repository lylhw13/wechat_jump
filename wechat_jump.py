# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os
import time
import random

X_train = np.loadtxt('data/X_train_1440_2560.txt', np.float32)
y_train = np.loadtxt('data/y_train_1440_2560.txt', np.float32)
y_train = y_train.reshape((y_train.size,1))

knn_model = cv2.ml.KNearest_create()
knn_model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

resolution = 15


def recog_score(img):
    rows, cols, channels = img.shape
    img_num = img[:int(0.3 * rows), 0: int(0.5 * cols)]

    out = np.zeros(img_num.shape, np.uint8)
    img_num = cv2.GaussianBlur(img_num, (5, 5), 0)
    gray = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    _, cons, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img_num, cons, -1, (0, 0, 255), 2)
    rects = np.array([cv2.boundingRect(con) for con in cons])
    w_max = max(rects[:, 2])
    h_max = max(rects[:, 3])
    pos_nums = []
    for rect in rects:
        x, y, w, h = rect
        if w > 0.95 * w_max or h > 0.95 * h_max:
            #cv2.rectangle(img, (x, y), (x + w_max, y + h_max), (0, 255, 0), 2)
            modify_w = w_max if h_max < 2 * w_max else 0.79 * h_max  # if the number 11 will cause error
            roi = thresh[y:y + h_max, x:x + int(modify_w)]
            roismall = cv2.resize(roi, (resolution, resolution))
            roismall = roismall.reshape((1, resolution * resolution))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = knn_model.findNearest(roismall, k=1)
            # np.append(pos_nums, (x, results[0][0]))
            pos_nums.append([x, results[0][0]])
            string = str(int((results[0][0])))
            cv2.putText(out, string, (x, y + h_max), 0, 1, (0, 0, 255))

    pos_nums = np.array(pos_nums)  # convert list to numpy.array
    res = 0
    for i in pos_nums[:, 0].argsort():
        x = pos_nums[i][1]
        res = res * 10 + x
    return int(res)


def cal_distance(img):
    rows, cols, _ = img.shape
    # calculate the chess position
    chess = cv2.imread('D:/test/chess_1440_2560.jpg')
    h, w, _ = chess.shape
    res = cv2.matchTemplate(img, chess, cv2.TM_CCOEFF_NORMED)
    *_, max_loc = cv2.minMaxLoc(res)

    chess_top_left = max_loc
    chess_bottom_right = (chess_top_left[0] + w, chess_top_left[1] + h)
    cv2.rectangle(img, chess_top_left, chess_bottom_right, (0, 0, 255))

    chess_position = (chess_top_left[0] + int(0.5*w),chess_bottom_right[1])

    # calculate the target position
    rows, cols, _ = img.shape
    height = img.shape[0]
    width = img.shape[1]

    img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_temp1 = cv2.GaussianBlur(img_temp, (5, 5), 0)
    img_gray = cv2.Canny(img_temp1, 20, 80)

    target = (0, 0)

    for y in range(int(0.3 * height), int(0.5 * height)):
        flag = False
        for x in range(width):
            if chess_top_left[0] - 0.2*w <= x <= chess_bottom_right[0] + 0.2*w:  # the condition that the chess is higher than the target
                continue
            #cv2.circle(img,(x,y),3,(0,255,0),1)
            if img_gray.item(y, x) == 255:
                flag = True
                #print(img_gray.item(y, x))
                target = (x, y)
                #print(target)
                break
        if flag:
            break

    # record every position
    # the top position
    cv2.circle(img, target, 10, (0, 0, 255), 2)
    # draw the chess position
    cv2.circle(img, chess_position, 10, (0, 255, 0), 2)
    # final position
    final_position = (target[0],chess_position[1] - int(0.85 * (chess_position[1] - target[1])))
    cv2.circle(img, final_position, 10, (255,0,0), 2)

    distance = np.linalg.norm(np.array(final_position) - np.array(chess_position))
    return distance


def detect_failure(img):
    play_again_img = cv2.imread('D:/test/play_again_1440_2560.jpg')

    res = cv2.matchTemplate(img, play_again_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    return True if max_val > 0.95 else False


def pull_pic(curr_dir):
    cmd = 'adb -s 192.168.32.101:5555 shell screencap -p /sdcard/jump.jpg'
    os.system(cmd)
    cmd = 'adb -s 192.168.32.101:5555 pull /sdcard/jump.jpg ' + curr_dir
    os.system(cmd)


work_dir = './image/'
goal = 10
time_ratio = 1  # for the 1440*2560 the ratio is 1


def main():
    global goal
    while True:
        pull_pic(work_dir)
        file = work_dir + 'jump.jpg'
        img = cv2.imread(file)

        # if detect_failure(img):
        #     print('Game over!')
        #     break

        rows, cols, _ = img.shape
        current_score = recog_score(img)
        print('*** current score is {} ***'.format(str(current_score)))

        dis = cal_distance(img)

        # record every step
        os.rename(file, work_dir + str(current_score) + '.jpg')
        cv2.imwrite(work_dir + str(current_score) + '-mark.jpg', img)

        press_time = dis * time_ratio

        if current_score >= goal:
            print('*' * 50)
            prompt = 'Do you want to exit the game?\n y/n?'
            answer = input(prompt)

            if answer.startswith('n'):
                prompt = 'Set your new score(new score must be bigger than current score):\n'
                goal = int(input(prompt))
            else:
                press_time = 2000       #exit the game

            print('*' * 50)

        cmd = 'adb -s 192.168.32.101:5555 shell input swipe {0} {1} {2} {3} {4}'.format(
            int(cols * 0.5) + random.randint(0, 150), int(rows * 0.75) + random.randint(0, 150),
            int(cols * 0.5) + random.randint(0, 150), int(rows * 0.75) + random.randint(0, 150),
            int(press_time)
        )
        os.system(cmd)
        time.sleep(1.3 + random.random())

        if press_time == 2000:
            break


if __name__ == '__main__':
    main()