import numpy as np
import cv2
import math
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook


cap = cv2.VideoCapture(0)#カメラ番号があっているか
print('camera is opened', cap.isOpened())
cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
#WidthxHeightの設定が正しいか
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # カメラ画像の横幅を640に設定
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320) # カメラ画像の縦幅を320に設定
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720) # カメラ画像の横幅を720に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を480に設定

#グリット線の初期値の縦線と横線
x = 30
y = 30

#マーカの実半径
r = 10

# 閾値 0-255
threshold_value = 50

showImgType = 0
lastShowImgType = 0
showTitle = 'Original image'

measurement = False

#各要素のbufferを作成
buffer_t = []
buffer_u = []
buffer_v = []
buffer_rdot = []
buffer_X = []
buffer_Y = []
buffer_Z = []

while(True):
    key = cv2.waitKey(1) & 0xFF


    #qをクリックすると停止
    if key == ord('q'):
        break

    #tをクリックすると画面が切り替わる
    elif key == ord('t'):
        showImgType = (showImgType + 1) % 3

    #gを押すとmeasurementが切り替わる
    elif key == ord('g'):
        print('g  clicked')
        if measurement == False:
            measurement = True
            start = time.time()
        elif measurement == True:
            measurement = False


    #sをクリックするとX,Y,Zをcsvファイルに書き込む
    elif key == ord('s'):

        print('start saving')
        with open('result.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['経過時間t[ms]','位置u[px]','位置v[px]','マーカーの半径r_dot[px]','X','Y','Z'])
            for i in range(len(buffer_t)):
                writer.writerow([buffer_t[i],buffer_u[i],buffer_v[i],buffer_rdot[i],buffer_X[i],buffer_Y[i],buffer_Z[i]])
        #save t, x, y, z to a csv file
        print('saving finished')

     #rをクリックするとcsvfileを読み込んで各値と三次元空間上の運動の軌道を出力
    elif key == ord('r'):
        df = pd.read_csv('result.csv')
        print(df)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter( df["X"], df["Y"], df["Z"],s=50, c="r",marker="o", alpha=0.5)
        # グラフの装飾
        ax.set_title("三次元上の軌道",fontsize=15) # タイトル
        ax.set_xlabel("X", fontsize=10) # x軸ラベル
        ax.set_ylabel("Y", fontsize=10) # y軸ラベル
        ax.set_zlabel("Z", fontsize=10) # z軸ラベル
        ax.view_init(30, 140) # 3Dの表示角度


    #pをクリックする度に閾値が5ずつ増加していく
    elif key == ord('p'):
        threshold_value += 5
        print(threshold_value)

    #mをクリックする度に閾値が5ずつ減少していく
    elif key == ord('m'):
        threshold_value -= 5
        print(threshold_value)

    #閾値が0または255に達すると停止
    elif threshold_value< 0 or threshold_value >255:
        break

    #グリット線の移動の処理
    elif key == ord('1'):
        if x == 640:
            print('max x 640')
        else:
                x += 10
                cv2.line(show_img,(0,x),(640,x),(255,0,0),1)
    elif key == ord('2'):
        if x == 0:
            print('min x 0')
        else:
                x -=  10
                cv2.line(show_img,(0,x),(640,x),(255,0,0),1)
    elif key == ord('3'):
        if y == 320:
            print('max y 320')
        else:
            y += 10
            cv2.line(show_img,(y,0),(y,320),(255,0,0),1)
    elif key == ord('4'):
        if y == 0:
            print('min y 0')
        else:
            y -= 10
            cv2.line(show_img,(y,0),(y,320),(255,0,0),1)
    #fをクリックするとその時のX,Y,Zの座標が出力される
    elif key == ord('f'):
        print(X,Y,Z)

    ret, frame = cap.read()
    #グレースケールに変換
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #出力画像配列の作成
    #threshold_img = grayFrame.copy()
    #グレースケールを2値化画像に変換
    #threshold_img[grayFrame < threshold_value] = 255
    #threshold_img[grayFrame >= threshold_value] = 0

    th, im_th = cv2.threshold(grayFrame, threshold_value, 255, cv2.THRESH_BINARY)
    #th, im_th = cv2.threshold(grayFrame, 0, 255, cv2.THRESH_OTSU)
    #print(th)

    threshold_img = cv2.bitwise_not(im_th)
    contours,hierarchy = cv2.findContours(threshold_img, 1, 2)

    u, v, s, r_dot = 0, 0, 0.0, 0.0
    if len(contours) > 0:
        for cnt in contours:
            #入力画像のモーメント
            #mu = cv2.moments(threshold_img, False)
            mu = cv2.moments(cnt)
            #モーメントからu,v座標を計算
            if mu["m00"] > 0:
                if cv2.contourArea(cnt) > 400:
                    u,v= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
                    s = cv2.contourArea(cnt)#面積
                    r_dot = round(math.sqrt(s/math.pi), 2)#半径

    #TODO (x, y, zの計算）

    L =310
    fx =662
    fy = 662
    f = 662

    #x, y, z 計算

    uz = u-320
    vz = -v+240

    if r_dot == 0.0: #r_dot=0の場合に処理が止まってしまうの(フリーズ)を防ぐ
        X = 0
        Y = 0
        Z = 0
    else:
        Z = f * r / r_dot
        X=(Z/fx) * uz
        Y=(Z/fy) * vz



    #TODO (bufferにデータを保存)
    if measurement == True:
        #t, u, vをあるバッファに保存
        t = time.time()
        tms = (t-start)*1000
        buffer_t.append(tms)
        buffer_u.append(u)
        buffer_v.append(v)
        buffer_rdot.append(r_dot)

        buffer_X.append(X)
        buffer_Y.append(Y)
        buffer_Z.append(Z)

    if lastShowImgType is not showImgType:
        cv2.destroyWindow(showTitle)
    lastShowImgType = showImgType

    if showImgType == 0:
        show_img = frame.copy()
        showTitle = 'Original image'
        r_, g_, b_ = 0, 200, 0

    elif showImgType == 1:
        show_img = grayFrame.copy()
        showTitle = 'gray image'
        r_, g_, b_ = 255, 0, 0

    elif showImgType == 2:
        show_img = threshold_img.copy()
        showTitle = 'binary image'
        r_, g_, b_ = 0, 0, 255

    #draw

    cv2.line(show_img,(0,x),(640,x),(255,0,0),1)
    cv2.line(show_img,(y,0),(y,320),(255,0,0),1)
    cv2.circle(show_img, (u,v), 6, (r_, g_, b_), -1)
    text = str(u) + ', ' + str(v) + ', ' + str(r_dot) + ', c: ' + str(len(contours))
    #text = str(x) + ', ' + str(y) + ', ' + str(z)
    cv2.putText(show_img, text, (u + 20, v+ 20),
               cv2.FONT_HERSHEY_PLAIN, 1.0,
               (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(showTitle, show_img)


cap.release()
cv2.destroyAllWindows()
