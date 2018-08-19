import cv2
import numpy as np
import functools
from matplotlib import pyplot as plt


def detectLines(img):
    minLineLength = 4
    maxLineGap = 100
    lines = cv2.HoughLinesP(img_edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print(lines)

# 直交座標系 → 極座標系
def convertPort(tpl):
    x = tpl[0]
    y = tpl[1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return tuple((r, theta))

# 極座標表示
def showPortGraph(vec):
    ax2 = plt.subplot(1,2,1,polar=True)
    for v in vec:
        ax2.scatter(v[0],v[1])
    ax2.set_rmax(4)
    ax2.grid(True)
    plt.show()

def showGraph(vec):
    plt.subplot(111),plt.imshow(img_edges)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    ax1 = plt.subplot(211)
    for i in range(len(vec)):
        ax1.scatter(i, vec[i][1])
    ax1.axis([0,len(vec),-4,4])
    ax1.grid(True)
    ax2 = plt.subplot(212)
    for i in range(len(vec)):
        ax2.scatter(i,abs(vec[i][1]))
    ax2.axis([0,len(vec),0,4])
    ax2.grid(True)
    plt.show()

img_origin = cv2.imread('img/white_0_1.jpg')
hw = img_origin.shape[0] / img_origin.shape[1]
img_half = cv2.resize(img_origin, (200, int(200.0*hw)))
img_gray = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (81,81), sigmaX=1)
img_edges = cv2.Canny(img_gauss,50,200)

#img_edges_rev = cv2.bitwise_not(img_edges)
#img_ad = cv2.threshold(img_gauss, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
#img_add = cv2.addWeighted(img_ad,0.3,img_edges,0.3,0)
# img_ad  = cv2.adaptiveThreshold(img_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3);

# 点検出
# SIMPLE:中間点保持しない NONE:保持する
contours = (cv2.findContours(img_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1])[0]


maxX=0
minX=200
maxY=0
minY=200*hw
for cnt in contours:
    if maxX < cnt[0][0]:
        maxX = cnt[0][0]
    if minX > cnt[0][0]:
        minX = cnt[0][0]
    if maxY < cnt[0][1]:
        maxY = cnt[0][1]
    if minY > cnt[0][1]:
        minY = cnt[0][1]

#cv2.line(img_edges, (minX, maxY), (maxX, maxY), (50, 0, 100), 2)
#cv2.line(img_edges, (minX, minY), (maxX, minY), (50, 0, 100), 2)
#cv2.line(img_edges, (minX, maxY), (minX, minY), (50, 0, 100), 2)
#cv2.line(img_edges, (maxX, minY), (maxX, maxY), (50, 0, 100), 2)
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

rect = cv2.minAreaRect(contours)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img_edges,[box],0,(50,0,100),2)

# 20, 35, 80, 114
#cv2.drawMarker(img_edges, tuple(contours[30][0]), (50,0,100), markerSize=10)
#cv2.drawMarker(img_edges, tuple(contours[40][0]), (50,0,100), markerSize=10)
#cv2.drawMarker(img_edges, tuple(contours[113][0]), (50,0,100), markerSize=10)
#cv2.drawMarker(img_edges, tuple(contours[114][0]), (50,0,100), markerSize=10)

vecLst = []
for i in range(len(contours)-1):
    vecLst.append(tuple(contours[i+1][0] - contours[i][0]))

vecPort = list(map(convertPort, vecLst))

showGraph(vecPort)
#plt.subplot(121),plt.imshow(img_gray)
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#plt.subplot(122),plt.imshow(img_edges)
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()
