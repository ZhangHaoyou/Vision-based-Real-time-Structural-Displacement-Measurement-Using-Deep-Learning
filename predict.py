#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

def addMask(image, roi, crop=True, crop_size=(640,640)):
    img = np.array(image)
    xmin, ymin, xmax, ymax = roi[0], roi[1], roi[2], roi[3]
    roiImg = img[ymin:ymax, xmin:xmax].copy()

    img[:,:,:] = 0
    img[ymin:ymax, xmin:xmax] = roiImg
    # print(img.shape) # (1080, 1920, 3) (height, width, channels)

    if crop:
        xc = (xmin+xmax)//2
        yc = (ymin+ymax)//2
        # img = img[yc-crop_size[1]//2:yc+crop_size[1]//2, xmin:xmin+crop_size[0], :] # T2
        # img = img[ymin:ymin+crop_size[1], xc-crop_size[0]//2:xc+crop_size[0]//2, :] # M45 # EQ1
        # img = img[ymax-crop_size[1]:ymax, xc-crop_size[0]//2:xc+crop_size[0]//2, :] # M84 # M41 # M51 # B1
        img = img[yc-crop_size[1]//2:yc+crop_size[1]//2, xc-crop_size[0]//2:xc+crop_size[0]//2, :] # SW1_M41
        # img = img[yc-crop_size[1]//2:yc+crop_size[1]//2, xc-crop_size[0]//2:xc+crop_size[0]//2, :] # B5
        # if yc-480 < 0:
        #     img = img[ 0:960, xc-480:xc+480, :]
        # else:
        #     img = img[yc-480:yc+480, xc-480:xc+480, :]

    return Image.fromarray(img)


if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = "yyy.mp4"
    video_fps       = 30.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "frames/"
    dir_save_path   = "detected_SW1_B2/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    with open("./logs/position.txt", "w") as txt:
            txt.write("time\tNum\ttop\tleft\tbottom\tright\n")

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        
        while 1:
            capture = cv2.VideoCapture(video_path)
            capture.set(3,1920) # width
            capture.set(4,1080) # height
            # capture.set(10, 70) # brightness
            # capture.set(11, 75) # contrast
            # capture.set(12, 50)
            ref, frame = capture.read()
            if ref == False:
                video_path += 1
            elif video_path == 1000000 or ref == True:
                print(video_path)
                break

        print("Camera is opened using path ID ", video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        fnum = 0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            fnum += 1
            # frame = cv2.resize(frame, (1920, 1080))
            cv2.imwrite("./img/frame_"+str(fnum)+".jpg", frame)
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        #             M41-SW1  M84     B5     B1    M51   M41   M45   T2    EQ1
        startx = 1230 # 550  # 540  # 1370 # 1350 # 840 # 945 # 940 # 50  # 760
        starty = 775  # 300  # 520  # 310  # 800  # 820 # 810 # 270 # 500 # 100
        w      = 120  # 200  # 150  # 210  # 230  # 170 # 180 # 170 # 170 # 380
        h      = 75   # 170  # 150  # 200  # 250  # 170 # 190 # 200 # 300 # 160
        roi = [startx, starty, startx+w, starty+h]

        # times = open("./times.txt", "r")
        # t = times.readlines()

        count_ = 0
        img_names = os.listdir(dir_origin_path)
        img_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        for img_name in tqdm(img_names):
            count_ += 1
            # if int(t[count_-1].split("\t")[2]) == 1:
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                image       = addMask(image, roi, crop_size=(640,640), crop=True)
                r_image     = yolo.detect_image(image, num=count_)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
        # times.close()

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
