from validation import Validator
from yolo import YOLO
from plot_template import plot_template
from utils import correct_perspective

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from PIL import Image


class Depth_Estimator(Validator):
    
    def __init__(self, experiment_name='sine005_80',
                raw_data_folder='./experimental_data/raw_data/'):
        super().__init__()
        self.experiment_name = experiment_name
        self.raw_data_folder = raw_data_folder
        self.log_depth_folder = './logs/depth'
        self.front_camera_folder = './experimental_data/front_camera'
        self.side_camera_folder = './experimental_data/side_camera'
        self.detections_folder = './detections'
    
    def _move_to_center(self, data):
        minimum, maximum = np.min(data), np.max(data)
        center = (maximum + minimum)/2
        data = data - center
        return data
    
    def read_sensor_data(self):
        data_path = self.raw_data_folder + self.experiment_name + '.txt'
        sampling_frequency = 1000
        # AI 6	Accel Y	Accel X	AI 8	AI 9	Position X	Position Y	AI 5	AI 7
        data = np.loadtxt(data_path, skiprows=4)
        data = data[:, 3:7]
        data = data - data[1, :]
        disp8, disp9 = data[:, 0], data[:, 1]
        position_x, position_y = data[:, 2], data[:, 3]
        disp8 = self._move_to_center(disp8)
        disp9 = self._move_to_center(disp9)
        position_x = self._move_to_center(position_x)
        
        rows, cols = data.shape
        times_sensor = np.arange(0, rows/sampling_frequency, 1/sampling_frequency)
        fig, ax = plt.subplots()
        ax.plot(times_sensor, disp8)
        ax.plot(times_sensor, disp9)
        ax.plot(times_sensor, position_x)
        ax.plot(times_sensor, position_y)
        ax.set_xlabel('Time (second)')
        ax.set_ylabel('Displacement (mm)')
        lengends = ['Disp 8', 'Disp 9', 'Positioin X', 'Position Y']
        ax.legend(lengends)
        fig.savefig(os.path.join(self.log_depth_folder, self.experiment_name+'.png'))
        sensor_data = np.column_stack((times_sensor, disp8, disp9, position_x))
        header = 'Time,\tdisp8,\tdisp9,\tPositionX'
        np.savetxt(os.path.join(self.log_depth_folder, self.experiment_name+'.txt'),
                sensor_data, delimiter=',', header=header)
        self.sensor_data = sensor_data
        return sensor_data
    
    def _pad_image_to_center(self, img, desired_size=960):
        """
        Pad the input image with black pixels to make it 960x960, 
        with the original image centered.
        
        Args:
        - img: The input image (as a numpy array).
        - desired_size: The target size for the padded image (default is 960x960).
        
        Returns:
        - padded_img: The padded image with the original image centered.
        """
        # Get the original image size
        original_h, original_w = img.shape[:2]
        
        # Calculate the padding for width and height
        pad_h = (desired_size - original_h) // 2
        pad_w = (desired_size - original_w) // 2
        
        # Ensure the final padded size is exactly 960x960 by adjusting any rounding issues
        top_pad = pad_h
        bottom_pad = desired_size - original_h - top_pad
        left_pad = pad_w
        right_pad = desired_size - original_w - left_pad

        # Use cv2.copyMakeBorder to pad the image
        padded_img = cv2.copyMakeBorder(
            img, 
            top_pad, bottom_pad, left_pad, right_pad, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]  # Black color
        )
        
        return padded_img
    
    def read_video(self, start=0, video_path=None, perspective=False):
        """ Detect positions of targets in each image in the images_folders
        
        Args:
            video_name (str):           The name of the original recorded video
            start (int):                The initial position of the motion
            
        Returns:
            numpy.array: two dimensional numpy array of the motion of one target in the video frame stream
        """
        yolo = YOLO() # load the proposed model
        if video_path is None:
            video_path = os.path.join(self.front_camera_folder, self.experiment_name+'_front.mp4')
        video_capture = cv2.VideoCapture(video_path)
        self.video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        # detections_path = self.log_depth_folder + '/' + self.experiment_name
        # if not os.path.exists(detections_path):
        #     os.makedirs(detections_path)
        
        time_stamps = []
        x_lengths = []
        y_lengths = []
        positions = []
        positions_side = np.loadtxt(os.path.join(self.log_depth_folder,
                'positions_'+self.experiment_name+'_side.txt'), delimiter=',')
        count = 0
        while True:
            if count >= 2816:
                break
            ret, frame = video_capture.read()
            if not ret:
                break
            time_stamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            time_stamps.append(time_stamp)
            
            if not perspective:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
            else:
                # cv2.imwrite('frame0.jpg', frame)
                # cv2.imshow('Frame', frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                print(count)
                
                top, left, bottom, right = positions_side[count]
                width = right - left
                height = bottom - top
                margin_left = int(1.8*width)
                margin_right = int(2*width)
                margin_top = int(2.1*height)
                margin_bottom = int(1.8*height)
                # print(int(top), int(bottom), int(left), int(right))
                debug = False
                if count == 2817:
                    debug = True
                
                frame = correct_perspective.correct_perspective(frame, count=count,
                            position=[int(top), int(bottom), int(left), int(right)],
                            margin=[margin_top, margin_bottom, margin_left, margin_right], debug=debug)
                frame = self._pad_image_to_center(frame)
                
            
            frame = Image.fromarray(np.uint8(frame))
            detected_img = np.array(yolo.detect_image(frame)) # RGB
            frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR) # BRG
            # cv2.imwrite(os.path.join(detections_path, 'frame_'+str(int(frame_number))+'.jpg'), detected_img)
            # cv2.imshow(video_name, detected_img)

            left, top, right, bottom = yolo.results[0][0][:4]
            x_length = np.abs(right - left)
            y_length = np.abs(bottom - top)
            x_lengths.append(x_length)
            y_lengths.append(y_length)
            positions.append([top, left, bottom, right])
            count += 1
            
        video_capture.release()
        cv2.destroyAllWindows()
        
        time_stamps = np.array(time_stamps[start:])
        x_lengths = np.array(x_lengths[start:])
        y_lengths = np.array(y_lengths[start:])
        xy_lengths = np.column_stack((time_stamps, x_lengths, y_lengths))
        header = 'time,\tx_length,\ty_length'
        np.savetxt(self.log_depth_folder+'/'+f'time_xy_lengths_{self.experiment_name}.txt',
                xy_lengths, delimiter=',', header=header)
        positions = np.array(positions)
        header = 'top,\tleft,\tbottom,\tright'
        np.savetxt(self.log_depth_folder+'/'+'positions_{}.txt'.format(self.experiment_name),
                positions, delimiter=',', header=header)
        
        self.xy_lengths = xy_lengths
        
        return xy_lengths
    
    def _estimate_distance(self, pixels, physical_legth=4.5, focal_length=1553): # 2165.33
        # unit: cm
        return physical_legth*focal_length/pixels*10 # cm to mm
    
    def _match_data(self, high_freq_data, low_freq_data, mean=True, direction=''):
        """ Resample high_frequency_data to match the number of low_frequency_data
        
        Args:
            recorded_disp (numpy.array):    Two dimensional numpy array of recorded displacement (time, displacement)
            detected_disp (numpy.array):    Two dimensional numpy array of detected displacement (time, displacement)

        Returns:
            (numpy.array2d): resampled data
        """
        print('Sampling and matching data...')
        # keep the number of high_frequency_data and low_frequency_data same
        sampled_data = np.zeros_like(low_freq_data)
        rows = low_freq_data.shape[0]
        for i, point in tqdm(enumerate(low_freq_data)):
            if i == 0:
                #                                  reference, front, back
                xn, yn = self._find_recorded_point(point, point, low_freq_data[i+1], high_freq_data, mean)
            elif i < rows-1:
                xn, yn = self._find_recorded_point(point, low_freq_data[i-1], low_freq_data[i+1], high_freq_data, mean)
            elif i == rows-1:
                xn, yn = self._find_recorded_point(point, low_freq_data[i-1], None, high_freq_data)
            else:
                raise ValueError("Error in finding the recorded point.")
            sampled_data[i] = xn, yn
        
        validation_data = np.hstack((low_freq_data, sampled_data))
        header = 'Recorded displacements (t, d) and then detected displacements (t, d)'
        np.savetxt(os.path.join(self.detected_disp_folder, 'sampled_data_'+self.experiment_name+direction+'.txt'),
                validation_data, header=header)
        self.plot_motions(low_freq_data, sampled_data, name='validation')
        return sampled_data
        
    
    def get_depth(self):
        xy_lengths = np.loadtxt(self.log_depth_folder+'/'+f'time_xy_lengths_{self.experiment_name}.txt',
                                delimiter=',')
        start_video = 460
        times_video = xy_lengths[:,0]
        times_video = times_video - times_video[start_video]
        x_lengths = xy_lengths[:,1]
        y_lengths = xy_lengths[:,2]
        depths = self._estimate_distance(x_lengths)
        depths = depths - depths[start_video]
        depths = depths - depths[370]
        times_video = times_video[start_video:start_video+800]
        depths = depths[start_video:start_video+800]
        
        sensor_data = np.loadtxt(os.path.join(self.log_depth_folder, self.experiment_name+'.txt'),
                delimiter=',')
        start_sensor = 0
        times_sensor = sensor_data[:,0] - sensor_data[start_sensor,0]
        disp_x = sensor_data[:,3]
        
        # fine-tuning
        times_sensor = times_sensor[::100]
        disp_x = disp_x[::100]
        times_sensor = times_sensor[16:]
        times_sensor = times_sensor[:401] - times_sensor[0]
        disp_x = disp_x[16:401+16]
        
        self.times_sensor = times_sensor
        self.disp_x = disp_x
        self.times_video = times_video
        self.depths = depths
        
        sensor_sampled_data = np.column_stack((times_sensor, disp_x))
        header = 'time (s)\tdisplacement (mm)'
        np.savetxt(os.path.join(self.log_depth_folder, self.experiment_name+'sensor_40s.txt'),
                sensor_sampled_data, header=header, delimiter=',')
        
        video_cut_data = np.column_stack((times_video, depths))
        video_sampled_data = self._match_data(video_cut_data, sensor_sampled_data)
        
        header = 'time (s)\tdisplacement (mm)'
        np.savetxt(os.path.join(self.log_depth_folder, self.experiment_name+'video_40s.txt'),
                video_sampled_data, header=header, delimiter=',')
        
        # calculate the rmse
        rmse = np.sqrt(mean_squared_error(sensor_sampled_data[:,1], video_sampled_data[:,1]))
        print(f'The rmse is {rmse}')
        
        # plot
        fig, ax = plt.subplots()
        ax.plot(times_sensor, disp_x)
        ax.plot(video_sampled_data[:,0], video_sampled_data[:,1])
        ax.set_xlabel('Time (second)')
        ax.set_ylabel('Displacement (mm)')
        lengends = ['sensor', 'YOLO']
        ax.legend(lengends)
        fig.savefig(os.path.join(self.log_depth_folder,
                                self.experiment_name+'_comparison.png'))
        
        plot_template([times_sensor, disp_x], [video_sampled_data[:,0], video_sampled_data[:,1]],
                    xlabel='Time (s)', ylabel='Displacement (mm)', legend_labels=['Sensor', 'YOLOP-WNMS'],
                    colors=['black', 'red'], markers=[None, 's'], markeverys=[None, 20],
                    xlim=(0,40), ylim=(-100,200), major_x_interval=10, major_y_interval=60)
        
        
        pass

def depth_estimator_front_camera():
    experiment_name = 'sine005_80'
    estimator = Depth_Estimator(experiment_name=experiment_name)
    # estimator.read_sensor_data()
    # estimator.read_video()

    xy_lengths = np.loadtxt(estimator.log_depth_folder+'/'+
                            f'time_xy_lengths_{estimator.experiment_name}.txt',
                                delimiter=',')
    start_video = 460
    times_video = xy_lengths[:,0]
    times_video = times_video - times_video[start_video]
    x_lengths = xy_lengths[:,1]
    y_lengths = xy_lengths[:,2]
    depths = estimator._estimate_distance(x_lengths)
    depths = depths - depths[start_video]
    depths = depths - depths[370]
    times_video = times_video[start_video:start_video+800]
    depths = depths[start_video:start_video+800]
    
    sensor_data = np.loadtxt(os.path.join(estimator.log_depth_folder,
                                        estimator.experiment_name+'.txt'),
            delimiter=',')
    start_sensor = 0
    times_sensor = sensor_data[:,0] - sensor_data[start_sensor,0]
    disp_x = sensor_data[:,3]
    
    # fine-tuning
    times_sensor = times_sensor[::100]
    disp_x = disp_x[::100]
    times_sensor = times_sensor[16:]
    times_sensor = times_sensor[:401] - times_sensor[0]
    disp_x = disp_x[16:401+16]
    
    sensor_sampled_data = np.column_stack((times_sensor, disp_x))
    header = 'time (s)\tdisplacement (mm)'
    np.savetxt(os.path.join(estimator.log_depth_folder, estimator.experiment_name+'sensor_40s.txt'),
            sensor_sampled_data, header=header, delimiter=',')
    
    video_cut_data = np.column_stack((times_video, depths))
    video_sampled_data = estimator._match_data(video_cut_data, sensor_sampled_data)
    
    header = 'time (s)\tdisplacement (mm)'
    np.savetxt(os.path.join(estimator.log_depth_folder, estimator.experiment_name+'video_40s.txt'),
            video_sampled_data, header=header, delimiter=',')
    
    # calculate the rmse
    rmse = np.sqrt(mean_squared_error(sensor_sampled_data[:,1], video_sampled_data[:,1]))
    print(f'The rmse is {rmse}')
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(times_sensor, disp_x)
    ax.plot(video_sampled_data[:,0], video_sampled_data[:,1])
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Displacement (mm)')
    lengends = ['sensor', 'YOLO']
    ax.legend(lengends)
    fig.savefig(os.path.join(estimator.log_depth_folder,
                            estimator.experiment_name+'_comparison.png'))
    
    plot_template([times_sensor, disp_x], [video_sampled_data[:,0], video_sampled_data[:,1]],
                xlabel='Time (s)', ylabel='Displacement (mm)', legend_labels=['Sensor', 'YOLOP-WNMS'],
                colors=['black', 'red'], markers=[None, 's'], markeverys=[None, 20],
                xlim=(0,40), ylim=(-100,200), major_x_interval=10, major_y_interval=60)

def depth_estimator_side_camera():
    experiment_name = 'sine005_80'
    estimator = Depth_Estimator(experiment_name=experiment_name)
    # video_path = os.path.join(estimator.side_camera_folder, estimator.experiment_name+'_side.mov')
    # estimator.read_video(video_path=video_path, perspective=True)

    xy_lengths = np.loadtxt(estimator.log_depth_folder+'/'+
                            f'time_xy_lengths_{estimator.experiment_name}.txt',
                                delimiter=',')
    start_video = 1265
    times_video = xy_lengths[:,0]
    times_video = times_video - times_video[start_video]
    x_lengths = xy_lengths[:,1]
    y_lengths = xy_lengths[:,2]
    depths = estimator._estimate_distance(x_lengths, focal_length=1500)
    depths = depths - depths[360]
    times_video = times_video[start_video:] # start_video+800
    depths = depths[start_video:] # start_video+800
    
    # fig, ax = plt.subplots()
    # ax.plot(times_video, depths)
    # plt.show()
    
    
    sensor_sampled_data = np.loadtxt(os.path.join(estimator.log_depth_folder, estimator.experiment_name+'sensor_40s.txt'),
            delimiter=',')
    
    video_cut_data = np.column_stack((times_video, depths))
    video_sampled_data = estimator._match_data(video_cut_data, sensor_sampled_data)
    
    header = 'time (s)\tdisplacement (mm)'
    np.savetxt(os.path.join(estimator.log_depth_folder, estimator.experiment_name+'video_40s.txt'),
            video_sampled_data, header=header, delimiter=',')
    
    # # calculate the rmse
    rmse = np.sqrt(mean_squared_error(sensor_sampled_data[:,1], video_sampled_data[:,1]))
    print(f'The rmse is {rmse}')
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(sensor_sampled_data[:,0], sensor_sampled_data[:,1])
    ax.plot(video_sampled_data[:,0], video_sampled_data[:,1])
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Displacement (mm)')
    lengends = ['sensor', 'YOLO']
    ax.legend(lengends)
    fig.savefig(os.path.join(estimator.log_depth_folder,
                            estimator.experiment_name+'_comparison.png'))
    
    # plot_template([times_sensor, disp_x], [video_sampled_data[:,0], video_sampled_data[:,1]],
    #             xlabel='Time (s)', ylabel='Displacement (mm)', legend_labels=['Sensor', 'YOLOP-WNMS'],
    #             colors=['black', 'red'], markers=[None, 's'], markeverys=[None, 20],
    #             xlim=(0,40), ylim=(-100,200), major_x_interval=10, major_y_interval=60)

    
def main():
    # depth_estimator_front_camera()
    sensor_front = np.loadtxt('logs/depth_front/sine005_80sensor_40s.txt', delimiter=',')
    video_front = np.loadtxt('logs/depth_front/sine005_80video_40s.txt', delimiter=',')
    video_side = np.loadtxt('logs/depth/sine005_80video_40s.txt', delimiter=',')
    plot_template([sensor_front[:,0], sensor_front[:,1]], [video_front[:,0], video_front[:,1]],
        [video_side[:,0], video_side[:,1]], figsize=(10,5),
        legend_labels=['Sensor', 'Frontal camera', 'Angled camera'],
        save_path='comparison.svg', xlabel='Time (s)', ylabel='Displacement (mm)',
        xlim=(0,40), ylim=(-100, 140), major_x_interval=10, major_y_interval=60,
        colors=['black', 'firebrick', 'blue'], linestyles=['-', '-', '-'],
        markers=['o', 's', 'd'], markeverys=[40, 50, 60], legend_horizontal=True,
        )
    # rmse = np.sqrt(mean_squared_error(sensor_front, video_front))
    # print(rmse)
    # depth_estimator_side_camera()

if __name__ == "__main__":
    main()