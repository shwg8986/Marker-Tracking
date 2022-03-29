import numpy as np
import cv2
import math
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook


#############
def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = (int)(Ax1 + uA * (Ax2 - Ax1))
    y = (int)(Ay1 + uA * (Ay2 - Ay1))

    return x, y
####
def TwoPointDistance(p1, p2):
    distance = (int)(math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)))
    return distance
####
cameraWidth = 640#720
cameraHeight =480#320
cap = cv2.VideoCapture(0)
print('camera is opened', cap.isOpened())
cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cameraWidth)  # カメラ画像の横幅を640に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraHeight) # カメラ画像の縦幅を320に設定


# 閾値 0-255
threshold_value = 50
L_th = 100
H_th = 300
hough_threshold=50

showImgType = 0
lastShowImgType = 0
showTitle = 'Original image'

measurement = False
initPoints = True
myptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor of previous loop


#各要素のbufferを作成
buffer_t = []

buffer_X = []
buffer_Y = []
buffer_Z = []

buffer_Roll = []
buffer_Pitch = []
buffer_Yaw = []

buffer_Roll_angle = []
buffer_Pitch_angle = []
buffer_Yaw_angle = []

buffer_P0 = []
buffer_P1 = []
buffer_P2 = []
buffer_P3 = []

while(True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('t'):
        showImgType = (showImgType + 1) % 4

    elif key == ord('g'):
        print('g  clicked')
        if measurement == False:
            measurement = True
            start = time.time()
        elif measurement == True:
            measurement = False

    #sをクリックするとcsvファイルに書き込む
    elif key == ord(','):
        print('start saving')
        with open('result2.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['経過時間t[ms]','X','Y','Z','Roll','Pitch','Yaw','Roll angle（度数)','Pitch angle(度数)','Yaw angle(度数)','P0(x,y,z)','P1(x,y,z)','P2(x,y,z)','P3(x,y,z)',])
            for i in range(len(buffer_t)):
                writer.writerow([buffer_t[i],buffer_X[i],buffer_Y[i],buffer_Z[i],buffer_Roll[i],buffer_Pitch[i],buffer_Yaw[i],buffer_Roll_angle[i],buffer_Pitch_angle[i],buffer_Yaw_angle[i],buffer_P0[i],buffer_P1[i],buffer_P2[i],buffer_P3[i]])
        #save t, x, y, z to a csv file
        print('saving finished')

     #rをクリックするとcsvfileを読み込んで三次元グラフ出力
    elif key == ord('.'):
        df = pd.read_csv('result2.csv')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df["X"], df["Y"], df["Z"],s=50, c="r",marker="o", alpha=0.5)
        # グラフの装飾
        ax.set_title("三次元上の軌道",fontsize=15) # タイトル
        ax.set_xlabel("X", fontsize=10) # x軸ラベル
        ax.set_ylabel("Y", fontsize=10) # y軸ラベル
        ax.set_zlabel("Z", fontsize=10) # z軸ラベル
        ax.view_init(30, 140) # 3Dの表示角度


    elif key == ord('r'):
        initPoints = True

    elif key == ord('s'):
        print('start saving')
        #save t, x, y, z to a csv file
        print('saving finished')

    #L_th と H_th の変更
    elif key == ord('1'):
        L_th += 10
        print('L_th = ', L_th)
        edge_img = cv2.Canny(mask, L_th, H_th)

    elif key == ord('2'):
        L_th -=  10
        print('L_th = ', + L_th)
        edge_img = cv2.Canny(mask, L_th, H_th)

    elif key == ord('3'):
        H_th += 10
        print('H_th = ',+ H_th)
        edge_img = cv2.Canny(mask, L_th, H_th)

    elif key == ord('4'):
        H_th -= 10
        print('H_th = ', + H_th)
        edge_img = cv2.Canny(mask, L_th, H_th)

    #hough_threshold の変更

    #pをクリックする度にhough_thresholdが5ずつ増加していく
    elif key == ord('p'):
        hough_threshold += 5
        print('hough_threshold = ', hough_threshold)

    #mをクリックする度にhough_thresholdが5ずつ減少していく
    elif key == ord('m'):
        hough_threshold -= 5
        print('hough_threshold = ', hough_threshold)

     #hough_thresholdが20または230に達すると停止
    elif hough_threshold< 20 or hough_threshold >230:
        break

    elif key == ord('^'):
        print(X)
        print(Y)
        print(Z)
        break

    #TODO (keyの操作)
    #
    #
    #
    #


    ret, frame = cap.read()
    #グレースケールに変換
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #グレースケールを2値化画像に変換
    #th, im_th = cv2.threshold(grayFrame, threshold_value, 255, cv2.THRESH_BINARY)
    th, im_th = cv2.threshold(grayFrame, 0, 255, cv2.THRESH_OTSU)
    #print(th)
    threshold_img = cv2.bitwise_not(im_th)
    contours,hierarchy = cv2.findContours(threshold_img, 1, 2)

    mask = np.zeros(threshold_img.shape,np.uint8)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        #入力画像のモーメント
        mu = cv2.moments(cnt)
        #モーメントからu,v座標を計算
        if mu["m00"] > 0:
            if cv2.contourArea(cnt) > 400:
                #u,v= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
                cv2.drawContours(mask,[cnt],0,255,-1)

    edge_img = cv2.Canny(mask, L_th, H_th)
     #  Standard Hough Line Transform
    lines = cv2.HoughLines(edge_img, 1, np.pi / 180, hough_threshold, None, 0, 0)
    lnLines = 0
    ptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor
    nPtuv = [0 for m in range(4)] #number of nearest points of each edge point

    # Draw the lines
    if (lines is not None):
        nLines = len(lines)
        pt1 = [[0 for m in range(2)] for n in range(nLines)]
        pt2 = [[0 for m in range(2)] for n in range(nLines)]
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1[i] = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2[i] = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1[i], pt2[i], (0,0,255), 1, cv2.LINE_AA)

            for j in range(0, i):
                x1, y1 = pt1[j]
                x2, y2 = pt2[j]
                x3, y3 = pt1[i]
                x4, y4 = pt2[i]
                vector1 = (x2 - x1), (y2 - y1)
                vector2 = (x4 - x3), (y4 - y3)
                unit_vector1 = vector1 / np.linalg.norm(vector1)
                unit_vector2 = vector2 / np.linalg.norm(vector2)
                dot_product = np.dot(unit_vector1, unit_vector2)
                if dot_product > 1:
                    dot_product = 1
                elif dot_product < -1:
                    dot_product = -1
                angle = np.arccos(dot_product) #angle in radian
                if math.degrees(angle) > 60 and math.degrees(angle) < 120:
                    pt = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
                    if pt is not None:
                        cv2.circle(frame, pt, 2, (255, 255, 255), -1)
                        _p = 0
                        while _p < 4:
                            pntCnt = 1
                            if nPtuv[_p] > 1:
                                pntCnt = nPtuv[_p]
                            ptCent = ptuv[_p][0] / pntCnt, ptuv[_p][1] / pntCnt
                            pntDis = TwoPointDistance(ptCent, pt)
                            if pntDis < 10 or nPtuv[_p] == 0: # the same group
                                ptuv[_p] = ptuv[_p][0] + pt[0], ptuv[_p][1] + pt[1]
                                nPtuv[_p] = nPtuv[_p] + 1
                                _p = 4
                            _p = _p + 1

    #print(nLines)

    nEdgePnt = 0
    for p in range(4):
        if nPtuv[p] != 0:
            nEdgePnt = nEdgePnt + 1
            ptuv[p] = (int)(ptuv[p][0] / nPtuv[p]), (int)(ptuv[p][1] / nPtuv[p])
            cv2.circle(frame, ptuv[p], 6, (255, 0, 0), 1)

    if nEdgePnt == 4:#4点のエッジを検出したら
        if initPoints == True:#点の順番のリセット
            initPoints = False
            initOrd = [0, 1, 2, 3]
            cw, ch = cameraWidth, cameraHeight
            originalPnt = [[0,ch],[0,0],[cw, 0], [cw, ch]]
            for ed in range(4):
                _ord = -1
                rp = 10000
                for p in range(4):
                    rx, ry = (ptuv[p][0] - originalPnt[ed][0]), (ptuv[p][1] - originalPnt[ed][1])
                    dr = math.sqrt(rx*rx + ry*ry)
                    if dr < rp:
                        rp = dr
                        _ord = p
                initOrd[ed] = _ord
                myptuv[ed] = ptuv[_ord]
            print(initOrd)
        _ptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor of previous loop
        #pntDist[0] = TwoPointDistance(ptuv[0], ptuv[1])
        rC = [-1, -1, -1, -1]
        for p in range(4):
            minDist = 10000
            for _p in range(4):
                if _p == rC[0] or _p == rC[1] or _p == rC[2] or _p == rC[3]:
                    continue
                dist_now_prev = TwoPointDistance(myptuv[p], ptuv[_p])
                if dist_now_prev < minDist:
                    minDist = dist_now_prev
                    _ptuv[p] = ptuv[_p]
                    rC[p] = _p


        ############　TODO　(x, y, z, roll, pitch, yawの計算)###########

        U = [0.0, 0.0, 0.0, 0.0]
        V = [0.0, 0.0, 0.0, 0.0]


        fx =  666
        fy = 658
        d = 60

        for p in range(4):
            myptuv[p] = _ptuv[p]
            a, b = _ptuv[p]
            cv2.putText(frame, ("p" + str(p) + str(myptuv[p])), (a + 20, b + 20),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 255), 1, cv2.LINE_AA)
            U[p] = (myptuv[p][0]-320)/fx
            V[p] = (-myptuv[p][1]+240)/fy

        ###p0, p1, p2, p3###
        x = [0, 0, 0, 0]
        y = [0, 0, 0, 0]
        z = [0, 0, 0, 0]
        i = [0, 0, 0]
        j = [0, 0, 0]
        k = [0, 0, 0]
        P = [0, 0, 0]

        if (U[1]*(V[2] -V[3])+U[2]*(V[3] -V[1])+U[3]*(V[1] -V[2])) == 0.0 or  ((U[2]*V[3]) - (U[3]*V[2])) == 0.0 : #0の場合に処理が止まってしまうの(フリーズ)を防ぐ
            X = 0.0
            Y = 0.0
            Z = 0.0
        else:
            A = (U[0]*(V[2] -V[3])+U[2]*(V[3] -V[0])+U[3]*(V[0] -V[2]))/(U[1]*(V[2] -V[3])+U[2]*(V[3] -V[1])+U[3]*(V[1] -V[2]))
            y[0] = d /(math.sqrt(((U[1]*A-U[0])**2)+((A-1)**2) + ((V[1]*A-V[0])**2)))
            y[1] = A*y[0]
            y[2] = (((U[1]*V[3]) - (U[3]*V[1]))* y[1] - ((U[0]*V[3]) - (U[3]*V[0]))*y[0])/((U[2]*V[3]) - (U[3]*V[2]))
            y[3] = (((U[1]*V[2]) - (U[2]*V[1]))* y[1] - ((U[0]*V[2]) -(U[2]*V[0]))* y[0])/((U[2]*V[3]) - (U[3]*V[2]))

            for p in range(4):
                x[p] = y[p]*U[p]
                z[p] = y[p]*V[p]

            X = (x[0]+x[1]+x[2]+x[3])/4
            Y = (y[0]+y[1]+y[2]+y[3])/4
            Z = (z[0]+z[1]+z[2]+z[3])/4

            Roll, Pitch, Yaw = 0, 0, 0
            i[0] = (1/d)*(x[3]-x[0])
            i[1] = (1/d)*(y[3]-y[0])
            i[2] = (1/d)*(z[3]-z[0])
            k[0] = (1/d) *(x[1]-x[0])
            k[1] = (1/d)*(y[1]-y[0])
            k[2] = (1/d)*(z[1]-z[0])
            j[0] = (1/ d**2)*((y[1]-y[0])*(z[3]-z[0])-(z[1]-z[0])*(y[3]-y[0]))
            j[1] = (1/ d**2)*((z[1]-z[0])*(x[3]-x[0])-(x[1]-x[0])*(z[3]-z[0]))
            j[2] = (1/ d**2)*((x[1]-x[0])*(y[3]-y[0])-(y[1]-y[0])*(x[3]-x[0]))

            P = [[i[0],j[0],k[0]],[i[1],j[1],k[1]],[i[2],j[2],k[2]]]


            Roll = math.atan2(j[2],k[2])
            Pitch = math.atan2(-i[2],math.sqrt(j[2]*j[2]+k[2]*k[2]))
            Yaw = math.atan2(i[1],i[0])

            #ラジアン(弧度法)から度数法に変換
            Roll_angle = (Roll/math.pi)*180
            Pitch_angle = (Pitch/math.pi)*180
            Yaw_angle = (Yaw/math.pi)*180

    #TODO (bufferにデータを保存)
    if measurement == True:
        #t, 各エッジ点（u, v）をあるバッファに保存
        t = time.time()
        tms = (t-start)*1000
        buffer_t.append(tms)
        buffer_X.append(X)
        buffer_Y.append(Y)
        buffer_Z.append(Z)

        buffer_Roll.append(Roll)
        buffer_Pitch.append(Pitch)
        buffer_Yaw.append(Yaw)

        buffer_Roll_angle.append(Roll_angle)
        buffer_Pitch_angle.append(Pitch_angle)
        buffer_Yaw_angle.append(Yaw_angle)


        buffer_P0.append((x[0],y[0],z[0]))
        buffer_P1.append((x[1],y[1],z[1]))
        buffer_P2.append((x[2],y[2],z[2]))
        buffer_P3.append((x[3],y[3],z[3]))




        # or save t, x, y, z, roll, pitch, yaw in a buffer when start Measement

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

    elif showImgType == 3:
        show_img = edge_img.copy()
        showTitle = 'edge image'
        r_, g_, b_ = 0, 0, 255

    cv2.imshow(showTitle, show_img)


cap.release()
cv2.destroyAllWindows()
