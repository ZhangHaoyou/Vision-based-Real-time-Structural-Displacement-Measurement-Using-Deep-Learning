import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def video2jpg(video_path='./video/DCVS2.mp4', interval=120, start=1, end=10e5):
    vc = cv2.VideoCapture(video_path)
    i = 1
    while True:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True and i%interval == 0 and start <= i <= end:
            cv2.imshow('video', frame)
            cv2.imwrite("./frames/frame_{}.jpg".format(i), frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        i += 1
        if i == end:
            break
    vc.release()
    cv2.destroyAllWindows()

def filterPhoto(img_path="./img", imgwalls_path="./img_walls", interval=12):
    img_names = os.listdir(imgwalls_path)
    img_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    frame_start = 0 # 44550
    for i, img_name in enumerate(tqdm(img_names)):
        if i % interval == 0:
            frame_num = int(img_name.split(".")[0].split("_")[-1])
            shutil.copy(os.path.join(imgwalls_path, img_name), os.path.join(img_path, "frame_"+str(frame_start+frame_num)+".jpg"))

class Position(object):

    def __init__(self, pos="./logs/position.txt", disp="./logs/displacement.txt") -> None:
        self.pos = open(pos, "r")
        self.disp = open(disp, "w")
        self.lines = self.pos.readlines()

    # def __enter__(self):

    def getCenter(self, target):
        target = target.split(",")
        return (float(target[1])+float(target[3]))/2, (float(target[0])+float(target[2]))/2
    
    def sortBottom2Top(self, xcs, ycs):
        # just for 2 target
        index_bottom = ycs.index(min(ycs))
        index_top = ycs.index(max(ycs))
        return [xcs[index_bottom], xcs[index_top]], [ycs[index_bottom], ycs[index_top]]
    
    def getNear(self,x, x1, x2):
        distance1 = (x1[0]-x[0])**2 + (x1[1]-x[1])**2
        distance2 = (x2[0]-x[0])**2 + (x2[1]-x[1])**2
        if distance1 < distance2:
            return x1
        else:
            return x2

    def pixel2disp(self, scale=1):
        
        start = self.lines[1].split()
        xc0s = []
        yc0s = []
        for target in start[2:]:
            xc0, yc0 = self.getCenter(target)
            xc0s.append(xc0)
            yc0s.append(yc0)
        xc0s, yc0s = self.sortBottom2Top(xc0s, yc0s)
        xc10, xc20 = xc0s[0], xc0s[1]
        yc10, yc20 = yc0s[0], yc0s[1]

        cs = []
        for l, line in enumerate(tqdm(self.lines[1:])):
            things = line.split()
            time = things[0] + " " + things[1] + ","
            xcs = []
            ycs = []
            for target in things[2:]:
                xc, yc = self.getCenter(target)
                xcs.append(xc)
                ycs.append(yc)
            if not xcs:
                self.disp.write(time+"\n")
                cs.append([])
            elif len(xcs) == 2:
                xcs, ycs = self.sortBottom2Top(xcs, ycs)
                xc1, xc2 = xcs[0], xcs[1]
                yc1, yc2 = ycs[0], ycs[1]
                x1, y1 = xc1-xc10, yc1-yc10
                x2, y2 = xc2-xc20, yc2-yc20
                self.disp.write(time+str(x1)+","+str(y1*scale)+","+str(x2*scale)+","+str(y2*scale)+"\n")
                cs.append([[xc1,yc1], [xc2,yc2]])
            elif len(xcs) == 1:
                i = 1
                while True:
                    c0 = cs[l-i]
                    if len(c0) !=2:
                        i += 1
                        continue
                    else:
                        near = self.getNear([xcs[0], ycs[0]], c0[0], c0[1])
                        x, y = xcs[0]-near[0], ycs[0]-near[1]
                        self.disp.write(time+str(x)+","+str(y)+"\n")
                        cs.append([[xcs[0], ycs[0]]])
                        break
            else:
                cs.append([])
                print("There is more than 2 targets in Line", l+1)
        
        def __exit__(self):
            self.pos.close()
            self.disp.close()

def modifyNum(txt_path, txt_new_path):
    with open(txt_path, "r") as txt:
        txt_new = open(txt_new_path, "w")
        txt_new.write("time\tNum\ttop\tleft\tbottom\tright\n")
        lines = txt.readlines()
        for i, l in enumerate(lines[1:]):
            parts = l.split("\t")
            parts[1] = str(i+1)
            txt_new.write("\t".join(parts))
        txt_new.close()

def readTime(name="./frames/frame_235002.jpg"):
    img = cv2.imread(name)[910:910+60, 90:90+210, :]
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([0, 43, 46])
    # upper_red = np.array([255, 255, 255])
    # # inRange()方法返回的矩阵只包含0,255 (CV_8U) 0表示不在区间内
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # img_red =  cv2.bitwise_and(img, img, mask=mask)
    # img_red = cv2.cvtColor(img_red, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(img_gray, 105, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((2,2), np.uint8)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, k)
    # cv2.imwrite("test.jpg", dst)
    text = pytesseract.image_to_string(dst)
    if len(text) != 9:
        print(name)
        cv2.imwrite("test.jpg", img)
    # print(text)
    return text

def getTimes(img_dir="./frames"):
    img_names = os.listdir(img_dir)
    img_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    last = "06:31:40\n"
    # with open("./times.txt", "r") as txt:
    #     start = len(txt.readlines())
    times = open("./times.txt", "w")
    start = 0
    end = 5000
    t = 0
    count = 0
    for img_name in tqdm(img_names):
        count += 1
        if  start < count < end:
            image_path  = os.path.join(img_dir, img_name)
            text = readTime(image_path)
            if text == last:
                times.write(text.replace("\n", "")+"\t"+str(t)+"\n")
            else:
                times.write(text.replace("\n", "")+"\t"+str(t+1)+"\n")
                last = text
                t += 1

def adjustFormat():
    times = open("./times_new.txt", "w")
    with open("./times.txt", "r") as txt:
        lines = txt.readlines()
        for i, l in enumerate(lines):
            if i%2 == 0:
                times.write(l.replace("\n", "") + lines[i+1])
    times.close()

def checkTime():
    time_new = open("time_new.txt", "w")
    counts = np.loadtxt("count.txt")
    with open("./times.txt", "r") as txt:
        lines = txt.readlines()
        sec = 0
        count = 0
        count_list = []
        last_s = 40
        last_m = 31
        last_h = 6
        for i, l in enumerate(lines):
            h = int(l[:2])
            m = int(l[3:5])
            s = int(l[6:8])
            # if s == last_s:
            #     pass
            # elif s - last_s == 1 or s-last_s == -59:
            #     last_s = s
            # else:
            #     print("Line ", i+1, l)
            #     break
            if s - last_s == 0:
                count += 1
            else:
                last_s = s
                count_list.append(count)
                count = 1
                sec += 1
            time_new.write(l[:8]+"\t"+str(round(sec+(count-1)/counts[sec], 3))+"\t"+str(count)+"\t"+str(int(counts[sec]))+"\n")
        count_list.append(count)
    time_new.close()
    # with open("./count.txt", "w") as txt:
    #     for c in count_list:
    #         txt.write(str(c)+"\n")
    print(sec)

def inZeros(x, zeros):
    for z in zeros:
        if z[0] < x < z[1]:
            return True
    return False

def readPosition(position_file, dictionary=False, threshold=5, zeros=[[0,0], [0,0]]):
    ycx = []
    ycy = []
    dict_xc = {}
    dict_yc = {}
    with open(position_file, "r") as txt:
        lines = txt.readlines()
        pos1 = lines[1].split("\t")[-2].split(",")
        xc0 = (float(pos1[1])+float(pos1[3]))/2
        yc0 = (float(pos1[0])+float(pos1[2]))/2
        for l in lines[1:]:
            pos = l.split("\t")[-2].split(",")
            frame = int(l.split("\t")[-3])
            if inZeros(frame, zeros):
                if dictionary:
                    dict_xc[frame] = 0
                    dict_yc[frame] = 0
                else:
                    ycx.append(0)
                    ycy.append(0)
            else:
                dx = (float(pos[1])+float(pos[3]))/2 - xc0
                dy = (float(pos[0])+float(pos[2]))/2 - yc0
                if dictionary:
                    dict_xc[frame] = (float(pos[1])+float(pos[3]))/2 if dx <= threshold else 0
                    dict_yc[frame] = (float(pos[0])+float(pos[2]))/2 if dy <= threshold else 0
                else:
                    if dx <= threshold:
                        ycx.append((float(pos[1])+float(pos[3]))/2)
                    else:
                        ycx.append(0)
                    if dy <= threshold:
                        ycy.append((float(pos[0])+float(pos[2]))/2)
                    else:
                        ycy.append(0)
    if dictionary:
        return dict_xc, dict_yc
    else:
        return ycx, ycy

def getNearPoint(p, data):
    for i, d in enumerate(data):
        if d <= p and p < data[i+1]:
            if p-d > data[i+1]-p:
                return i, data[i+1]
            else:
                return i, d

def getMean(data):
    count_zero = 0
    for d in data:
        if d == 0:
            count_zero += 1
    if count_zero == len(data):
        return 0
    else:
        return sum(data)/(len(data)-count_zero)

def getMotion(position, start=0, scale=1):
    motion = []
    if type(position) == type({"key":"value"}):
        for p in position:
            v = position[p]
            if v != 0:
                motion.append((v-position[start+1])*scale)
            else:
                motion.append(0)
    else:
        for p in position:
            if p != 0:
                motion.append((p-position[start])*scale)
            else:
                motion.append(0)
    return motion

def getOriginalPosition(x, y, roi, crop_size=(640,640)):
    if x == 0 or y == 0:
        return 0, 0
    else:
        startx = roi[0]
        starty = roi[1]
        w = roi[2]
        h = roi[3]
        xmin_roi = startx
        xmax_roi = startx + w
        ymin_roi = starty
        ymax_roi = starty + h
        xc_roi = (xmin_roi+xmax_roi)//2
        yc_roi = (ymin_roi+ymax_roi)//2
        xmin_img = xc_roi-crop_size[0]//2
        ymin_img = yc_roi-crop_size[1]//2
        return x+xmin_img, y+ymin_img

def getShearDisp(x1c_d, y1c_d, x2c_d, y2c_d, p1=[550,300,200,170], p2=[1230,775,120,75], start=0, scale=1):
    length = []
    for i in range(len(x1c_d)): # o means original, d mean detected
        x1c_o, y1c_o = getOriginalPosition(x1c_d[i], y1c_d[i], roi=p1, crop_size=(640,640))
        x2c_o, y2c_o = getOriginalPosition(x2c_d[i], y2c_d[i], roi=p2, crop_size=[640,640])
        if x2c_o != 0:
            x2c_o, y2c_o = x2c_o+(1250-1291), y2c_o+(715-815)
        length.append(((x1c_o-x2c_o)**2+(y1c_o-y2c_o)**2)**0.5)
    return getMotion(length, start=start, scale=scale)

def validatePRCW2():
    # disp22 = np.loadtxt("./logs/disp_PRCW2_D22.txt")
    disp9 = np.loadtxt("./logs/disp_PRCW2_D9.txt")-0.64
    times = open("./logs/times_PRCW2_video.txt", "r")
    time_lines = times.readlines()

    # _, yc_t2_ = readPosition("./logs/position_T2_.txt")
    # _, yc_t2  = readPosition("./logs/position_T2.txt")
    zeros = [[4682,6514], [8065,8349]]
    xc_m41, _ = readPosition("./logs/position_M41.txt", dictionary=True, threshold=5, zeros=zeros)
    xc_b1,  _ = readPosition("./logs/position_B1.txt", dictionary=True, threshold=5, zeros=zeros)

    temp_m41 = []
    temp_b1 = []
    # yc_t2_mean = []
    xc_m41_mean = []
    xc_b1_mean = []
    for i, l in enumerate(time_lines):
        t = float(l.split("\t")[1])
        count = int(l.split("\t")[2])
        frame = int(l.split("\t")[3])
        if i in xc_m41 and not inZeros(i+1,zeros):
            temp_m41.append(xc_m41[i])
        else:
            temp_m41.append(0)
        if i in xc_b1 and not inZeros(i+1,zeros):
            temp_b1.append(xc_b1[i])
        else:
            temp_b1.append(0)
        # temp.append(yc_t2[i])
        if count == frame:
            # yc_t2_mean.append(sum(temp)/len(temp))
            xc_m41_mean.append(getMean(temp_m41))
            xc_b1_mean.append(getMean(temp_b1))
            temp_m41 = []
            temp_b1 = []

            # shutil.copy("./detected_M41/frame_{}.png".format(32284+i), "./detected_M41_/frame_{}.png".format(32284+i))
    
    
    scale = 45/24 #*0.23 # + 1398/721)/2
    start = 3

    # disp_t2_ = (np.array(yc_t2_)-yc_t2_[8])*(-1*scale)
    # disp_t2_mean = (np.array(yc_t2_mean)-yc_t2_mean[7])*(-1*scale)
    disp_m41_mean = getMotion(xc_m41_mean, start=start, scale=scale)
    disp_b1_mean = getMotion(xc_b1_mean, start=start, scale=scale)
    dispy = np.array(disp_m41_mean)-np.array(disp_b1_mean)
    # plt.plot([i for i in range(len(disp22))], disp22, 'k')
    # plt.plot([i-8 for i in range(len(yc_t2_))], disp_t2_, 'r')
    # plt.plot([i-7 for i in range(len(yc_t2_mean))], disp_t2_mean, 'gray')
    plt.plot([i for i in range(len(disp9))], disp9, 'k')
    plt.plot([i-start for i in range(len(dispy))], -dispy, 'r')
    plt.show()

    # result = open("./logs/result_PRCW2.txt", "w")
    # result.write("time_sensor\tdisp_D22_sensor\tdisp_D22_detected_mean\tdisp_D22_detected_1\n")
    # for i, d in enumerate(disp22):
    #     if i+8 == len(yc_t2_):
    #         break
    #     else:
    #         result.write(str(i)+"\t"+str(d)+"\t"+str(disp_t2_mean[i+7])+"\t"+str(disp_t2_[i+8])+"\n")

    result = open("./logs/result_PRCW2_vertical.txt", "w")
    result.write("time_sensor\tdisp_D9_sensor\tdisp_D9_detected_mean\n")
    for i, d in enumerate(dispy):
        if i+start == len(dispy):
            break
        else:
            result.write(str(i)+"\t"+str(disp9[i])+"\t"+str(dispy[i+start])+"\n")
    
    times.close()
    result.close()

def validateSW1():
    start_index = 1924
    disp3 = np.loadtxt("./logs/disp_SW1_D3.txt")
    times = open("./logs/times_SW1_video.txt", "r")
    time_lines = times.readlines()

    zeros = [[1428,1500], [5400,6152], [8998,10321], [12051,12141], [13951,14491], [18600,19782]]

    xc_m41, yc_m41 = readPosition("./logs/position_SW1_M41.txt", dictionary=True, threshold=10, zeros=zeros)
    xc_b2, yc_b2   = readPosition("./logs/position_SW1_B2.txt", dictionary=True, threshold=10, zeros=zeros)

    temp_xc_m41 = []
    temp_yc_m41 = []
    temp_xc_b2  = []
    temp_yc_b2  = []
    xc_m41_mean = []
    yc_m41_mean = []
    xc_b2_mean  = []
    yc_b2_mean  = []
    for i, l in enumerate(time_lines):
        t = float(l.split("\t")[1])
        count = int(l.split("\t")[2])
        frame = int(l.split("\t")[3])
        if i in xc_m41 and i in xc_b2 and not inZeros(i+1,zeros): # i in xc_b2 and
            temp_xc_m41.append(xc_m41[i])
            temp_yc_m41.append(yc_m41[i])
            temp_xc_b2.append(xc_b2[i])
            temp_yc_b2.append(yc_b2[i])
        else:
            temp_xc_m41.append(0)
            temp_yc_m41.append(0)
            temp_xc_b2.append(0)
            temp_yc_b2.append(0)
        if count == frame:
            xc_m41_mean.append(getMean(temp_xc_m41))
            yc_m41_mean.append(getMean(temp_yc_m41))
            xc_b2_mean.append(getMean(temp_xc_b2))
            yc_b2_mean.append(getMean(temp_yc_b2))
            temp_xc_m41 = []
            temp_yc_m41 = []
            temp_xc_b2 = []
            temp_yc_b2 = []

            shutil.copy("./detected_SW1_M41/frame_{}.png".format(235002+i), "./detected_SW1_M41_/frame_{}.png".format(32284+i))

    scale_m41 = 45/26 #*0.23 # + 1398/721)/2
    scale_b2 = 45/29
    scale_shear = 1166.19/675.26#*0.42
    start = 55
    s = 1700
    disp = getShearDisp(xc_m41_mean[s:], yc_m41_mean[s:], xc_b2_mean[s:], yc_b2_mean[s:], start=start, scale=scale_shear)

    # dispx_m41 = getMotion(xc_m41_mean, start=start, scale=scale_m41)
    # dispy_m41 = getMotion(yc_m41_mean, start=start, scale=scale_m41)
    # disp_m41  = getShearDisp(dispx_m41, dispy_m41)
    # dispx_b2 = getMotion(xc_b2_mean, start=start, scale=scale_b2)
    # dispy_b2 = getMotion(yc_b2_mean, start=start, scale=scale_b2)
    # disp_b2  = getShearDisp(dispx_b2, dispy_b2)
    # # dispx = np.array(dispx_m41)-np.array(dispx_b2)
    # # dispy = np.array(dispy_m41)-np.array(dispy_b2)
    # # disp = getShearDisp(dispx, dispy)
    # disp = np.array(disp_m41) + np.array(disp_b2)
    
    plt.plot([i for i in range(len(disp3[1924:]))], disp3[1924:], 'k')
    plt.plot([i for i in range(len(disp[start:]))], disp[start:], 'r')
    # plt.plot([(i-start)*0.1 for i in range(len(disp))], disp, 'r')
    plt.show()

    result = open("./logs/result_SW1_shear.txt", "w")
    result.write("time_sensor\tdisp_D3_sensor\tdisp_D3_detected_mean\n")
    for i, d in enumerate(disp[start:]):
        result.write(str(i)+"\t"+str(disp3[1924:][i])+"\t"+str(disp[start:][i])+"\n")
    
    result.close()



def validateEQ1():
    # scale = 10*25.4/59
    # sensor_time = np.loadtxt("./logs/times_sensor_adjusted.txt")
    # video_time = np.loadtxt("./logs/timestamps_adjusted.txt")

    # sensor_disp = np.loadtxt("./logs/disp_sensor_EQ1.txt")
    # xc, _ = readPosition("./logs/position_EQ1_640.txt")
    # start = int(len(xc)-len(video_time))
    # video_disp = (np.array(xc) - xc[start])*scale
    # video_disp = video_disp[start:]

    # time_sensor_sampled = []
    # disp_sensor_sampled = []
    # result = open("./logs/result_EQ1.txt", "w")
    # result.write("time_sensor\tdisp_sensor\ttime_detected\tdisp_detected\n")
    # for index, t in enumerate(video_time):
    #     i, d = getNearPoint(t, sensor_time)
    #     time_sensor_sampled.append(d)
    #     disp_sensor_sampled.append(sensor_disp[i])
    #     result.write(str(d)+"\t"+str(sensor_disp[i])+"\t"+str(t)+"\t"+str(video_disp[index])+"\n")
    # result.close()

    data = np.loadtxt("./logs/result_EQ1.txt", skiprows=1)
    rmse = np.sqrt(mean_squared_error(data[195:1096,1], data[195:1096,3]))
    print(data[195,0], data[1095,0])
    print(rmse)

    # plt.plot(sensor_time, sensor_disp, "k")
    # plt.plot(time_sensor_sampled, disp_sensor_sampled, "gray")
    # plt.plot(video_time, video_disp, "r")
    plt.plot(data[:,0], data[:,1], "k")
    plt.plot(data[:,2], data[:,3], "r")
    plt.show()


if __name__ == "__main__":
    # pos = Position()
    # pos.pixel2disp(scale=90/45)
    # video2jpg(video_path="I:/2022-09-26-08-48-39.mp4", interval=1, start=235002, end=240002)
    # filterPhoto(interval=8)
    # modifyNum("./logs/position.txt", "./logs/position_T2.txt")
    # readTime()
    # getTimes()
    # adjustFormat()
    # checkTime()
    # validatePRCW2()
    validateSW1()
    # validateEQ1()
    pass