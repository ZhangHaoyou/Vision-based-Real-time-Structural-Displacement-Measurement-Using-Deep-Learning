from yolo import YOLO

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from PIL import Image


class Validator(object):
    
    """ Validation of the proposed YOLO-BWA-WNMS model using recorded videos of
            existing static and dynamic experiments
    
    Attributions:
        experiment_name (str):      The experiment name for finding and naming after folders and images
        video_folder (str):         The folder path of the recorder video
        images_folder (str):        A folder contains several sub-folders encompassing experimental video frame images
        detections_folder (str):    A folder for saving the image detections of every experiment validation
        disp_folder (str):          A folder containing the recorded displacement
        detected_disp_folder (str): A folder for saving the detected displacement and other important files related to it
        
    """
    
    def __init__(self, experiment_name='EQ1', video_folder='./video', images_folder='./images', detections_folder='./detections',
                 disp_folder='./logs/disp', detected_disp_folder='./logs/detected_disp'):
        self.experiment_name = experiment_name
        self.video_folder = video_folder
        self.images_folder = images_folder
        self.detections_folder = detections_folder
        self.disp_folder = disp_folder
        self.detected_disp_folder = detected_disp_folder
        
        if not os.path.exists(detections_folder):
            os.makedirs(detections_folder)
        if not os.path.exists(disp_folder):
            os.makedirs(disp_folder)
        if not os.path.exists(detected_disp_folder):
            os.makedirs(detected_disp_folder)
        
    def _sort_frames(self, frames_folder):
        """ Sort the frame names to follow the video time sequence
        
        Args:
            frames_folder (str): A folder path contains every frame image extracted from the recorded video
            
        Returns:
            list: a list of frame image names which follow the video time sequence
        
        """
        img_names = os.listdir(frames_folder)
        img_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1][-4:]))
        return img_names
    
    def _boxes2center(self, positions_file='./logs/positions.txt'):
        """ Read position text files, calculate the center position in two directions
        
        Args:
            positions_file (str): A file containing the detected target positions

        Returns:
            tuple (center_top_bottom, center_left_right): 
                A tuple of the center positions of detected targets in two directions
        """
        positions = []
        with open(positions_file, 'r') as pos:
            lines = pos.readlines()
        for line in lines[1:]:
            positions.append(line.split(',')[:-1])
        positions = np.array(positions, dtype=float)
        center_top_bottom = (positions[:,0] - positions[:,2])/2.0
        center_left_right = (positions[:,1] - positions[:,3])/2.0
        
        return center_top_bottom, center_left_right
    
    def _get_index(self, time, times):
        """ Get the nearest time index according to the reference time
        
        Args:
            time (float):   The reference time
            times (list):   A list of a time
        
        Returns:
            int: the index of the nearest time in the list of a time
        """
        for i in range(len(times)-1):
            if times[i] <= time <= times[i+1]:
                return i
            elif time <= times[0]:
                return 0
    
    def _remove_outliers(self, data, threshold=40):
        # Calculate the difference between consecutive elements
        diff = np.abs(np.diff(data))
        
        # Identify the indices where the difference is greater than the threshold
        outlier_indices = np.where(diff > threshold)[0]
        
        # Create a mask to keep only the non-outlier elements
        mask = np.ones(len(data), dtype=bool)
        
        # Remove the outliers by setting mask to False at the identified indices
        mask[outlier_indices] = False
        mask[outlier_indices + 1] = False  # Remove both the element and its neighbor
        # Filtered data
        filtered_data = data[mask]
        
        # Outliers (the ones to be removed)
        removed_data = data[~mask]

        # Print all data
        # print(f"Original data: {data}")
        # print(f"Filtered data: {filtered_data}")
        # print(f"Removed outliers: {removed_data}")
        
        # Return the filtered data
        return data[mask]
        
    
    def _find_recorded_point(self, reference, front, back, high_frequent_data, mean=False):
        """ Find the recorded point using the reference point and the front and back points
                by seeking in the high frequent data
        
        Args:
            reference (tuple):  a reference point (x, y)
            front (tuple):      a point front of the reference point (x0, y0)
            back (tuple):       a point back of the reference point (x1, y1)
            high_frequent_data (numpy.array): 
                                Two dimentional numpy array which has more points between the front and back points
        
        Returns:
            tuple: a tuple of the nearest point away from the reference point (xn, yn)
        """
        # cut a time between the front and back points
        # x means the time; y means the displacement
        x, y = reference
        x0, y0 = front
        if back is not None:
            x1, y1 = back
        else:
            x1, y1 = x0, y0
        xs = high_frequent_data[:,0]
        ys = high_frequent_data[:,1]
        front_index = self._get_index(x0, xs)
        back_index = self._get_index(x1, xs)
        # calculate the distance between the reference displacement and all the displacements from the front and back points
        distances = []
        for i in range(front_index, back_index+1):
            distance = np.abs(ys[i]-y)
            distances.append(distance)
        minimum_value_index = front_index + distances.index(min(distances))
        if mean:
            target_area_data = high_frequent_data[front_index:back_index, 1]
            y_mean = np.mean(self._remove_outliers(target_area_data))
            if type(float(y_mean)) != type(0.2):
                print(y_mean, reference, front, back)
            # print(xs[(front_index+back_index)//2], x0, x1)
            return xs[(front_index+back_index)//2], y_mean
        return xs[minimum_value_index], ys[minimum_value_index]
        
    def detect_images(self, experiment_name='EQ1', start=0):
        """ Detect positions of targets in each image in the images_folders
        
        Args:
            experiment_name (str):      The experiment name for finding and naming after folders and images
            start (int):                The initial position of the motion
            
        Returns:
            numpy.array: two dimensional numpy array of the motion of one target in the video frame stream
        """
        yolo = YOLO() # load the proposed model
        img_folder = self.images_folder + '/img_' + experiment_name
        img_names = self._sort_frames(img_folder)
        
        if not os.path.exists(self.detections_folder + '/' + experiment_name):
            os.makedirs(self.detections_folder + '/' + experiment_name)
            
        for img_name in tqdm(img_names):
            img_path  = os.path.join(img_folder, img_name)
            img       = Image.open(img_path)
            detected_img = yolo.detect_image(img)
            detected_img.save(self.detections_folder + '/' + experiment_name + '/d_' + img_name)
        center_x, center_y = self._boxes2center()
        disp_x, disp_y = center_x - center_x[start], center_y - center_y[start]
        
        return disp_x, disp_y
    
    def detect_video(self, video_name='EQ1_DVR8_South.mp4', start=0, mask_box=None, crop=True):
        """ Detect positions of targets in each image in the images_folders
        
        Args:
            video_name (str):           The name of the original recorded video
            start (int):                The initial position of the motion
            
        Returns:
            numpy.array: two dimensional numpy array of the motion of one target in the video frame stream
        """
        self.experiment_name = video_name.split('.')[0]
        
        yolo = YOLO() # load the proposed model
        
        video_path = os.path.join(self.video_folder, video_name)
        video_capture = cv2.VideoCapture(video_path)
        self.video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        detections_path = self.detections_folder + '/' + self.experiment_name
        if not os.path.exists(detections_path):
            os.makedirs(detections_path)
        
        time_stamps = []
        center_xs = []
        center_ys = []
        positions = []
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            time_stamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            time_stamps.append(time_stamp)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
            frame = Image.fromarray(np.uint8(frame))
            if mask_box is not None:
                black_frame = Image.new('RGB', frame.size, color=0)
                mask = Image.new('L', frame.size, color=0)
                mask.paste(255, box=mask_box)
                black_frame.paste(frame, mask=mask)
                frame = black_frame
            if crop:
                frame = frame.crop((0,100, 960,1060))
            detected_img = np.array(yolo.detect_image(frame)) # RGB
            frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR) # BRG
            cv2.imwrite(os.path.join(detections_path, 'frame_'+str(int(frame_number))+'.jpg'), detected_img)
            # cv2.imshow(video_name, detected_img)

            left, top, right, bottom = yolo.results[0][0][:4]
            center_left_right = (left+right)/2.0
            center_top_bottom = (top+bottom)/2.0
            center_xs.append(center_left_right)
            center_ys.append(center_top_bottom)
            positions.append([top, left, bottom, right])
            
            if frame_number > 3500:
                break
                 
        video_capture.release()
        cv2.destroyAllWindows()
        
        time_stamps = np.array(time_stamps[start:])
        center_xs = np.array(center_xs[start:])
        center_ys = np.array(center_ys[start:])
        center_xys = np.hstack((time_stamps.reshape(-1,1), center_xs.reshape(-1,1), center_ys.reshape(-1,1)))
        np.savetxt(self.detected_disp_folder+'/'+'time_center_xy_{}.txt'.format(self.experiment_name), center_xys)
        positions = np.array(positions)
        header = 'top\tleft\tbottom\tright\n'
        np.savetxt(self.detected_disp_folder+'/'+'positions_{}.txt'.format(self.experiment_name), positions, header=header)
        
        return center_xys
    
    def get_fps(self, video_name='EQ1_DVR8_South.mp4', start=0):
        """ Examine the frame per second of the model prediction
        
        Args:
            video_name (str):           The name of the original recorded video
            start (int):                The initial position of the motion
            
        Returns:
            float: an average fps
        """
        
        yolo = YOLO() # load the proposed model
        
        video_path = os.path.join(self.video_folder, video_name)
        video_capture = cv2.VideoCapture(video_path)
        self.video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        fps_list = []
        fps = 0.0
        while True:
            t1 = time.time()
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # BGR to RGB
            frame = Image.fromarray(np.uint8(frame))
            detected_img = np.array(yolo.detect_image(frame)) # RGB
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            fps_list.append(fps)

        video_capture.release()
        cv2.destroyAllWindows()
        
        np.savetxt('./logs/fps.txt', np.array(fps_list))
        average_fps = sum(fps_list)/len(fps_list)
        print('The average fps is', average_fps)
        
        return average_fps
    
    def plot_motions(self, *motions, name='temp'):
        """Plot the comparison of detected and recorded motions
        
        Args:
            detected_motion (list): A list of detected displacement
            recorded_motion (list): A list of recorded displacement
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        for motion in motions:
            ax.plot(motion[:,0], motion[:,1])
        # plt.show()
        fig.savefig(f'./logs/{name}.png')
        
    def calculate_error(self, recorded_disp, detected_disp, reverse=False, mean=False, direction=''):
        """ Calculate the root of mean squared error (rmse) between the recorded displacements and detected displacements
            p.s. the number of recorded_disp is larger than that of the detected_disp
        
        Args:
            recorded_disp (numpy.array):    Two dimensional numpy array of recorded displacement (time, displacement)
            detected_disp (numpy.array):    Two dimensional numpy array of detected displacement (time, displacement)
            reverse (bool):                 Control which displacements are saved in front two columns in the validation data

        Returns:
            float: the rmse
        """
        print('Calculating the rmse...')
        # keep the number of recorded_disp and detected_disp same
        sampled_recorded_disp = np.zeros_like(detected_disp)
        rows = detected_disp.shape[0]
        for i, point in tqdm(enumerate(detected_disp)):
            if i == 0:
                #                                  reference, front, back
                xn, yn = self._find_recorded_point(point, point, detected_disp[i+1], recorded_disp, mean)
            elif i < rows-1:
                xn, yn = self._find_recorded_point(point, detected_disp[i-1], detected_disp[i+1], recorded_disp, mean)
            elif i == rows-1:
                xn, yn = self._find_recorded_point(point, detected_disp[i-1], None, recorded_disp)
            else:
                raise ValueError("Error in finding the recorded point.")
            sampled_recorded_disp[i] = xn, yn
        if reverse:
            validation_data = np.hstack((detected_disp, sampled_recorded_disp))
        else:
            validation_data = np.hstack((sampled_recorded_disp, detected_disp))
        header = 'Recorded displacements (t, d) and then detected displacements (t, d)'
        np.savetxt(os.path.join(self.detected_disp_folder, 'sampled_data_'+self.experiment_name+direction+'.txt'),
                   validation_data, header=header)
        self.plot_motions(sampled_recorded_disp, detected_disp, name='validation')
        # calculate the rmse
        rmse = np.sqrt(mean_squared_error(sampled_recorded_disp[:,1], detected_disp[:,1]))
        print(f'The rmse is {rmse}')
        return rmse
        
        
def validate_EQ1(video_name='EQ1_DVR8_South.mp4', scale=10*25.4/59):
    """Validate the horizontal motion of the dynamic shaking table EQ1 experiment
        
        Args:
            video_name (str):           The name of the original recorded video
            scale (float):              The scale transfering units from pixel to mm
    """
    experiment_name = video_name.replace('.mp4', '')
    my_validator = Validator(experiment_name)

    # recorded data: Dyn1_node2_data.txt, first column is time (second)
    # the first row is titles; the second row is units, the third row is data 
    # sensor channel name: S2PF1 (Column 42, or -13); unit: inch
    # sensor_file_name = 'Dyn1_node2_data.txt'
    # sensor_data = np.loadtxt(os.path.join(my_validator.disp_folder, sensor_file_name), delimiter='\t', skiprows=2)
    # sensor_data = sensor_data[:, [0, 42]]
    # sensor_data[:, 1] = sensor_data[:, 1]*25.4 # inch to mm
    # rows = sensor_data.shape[0]
    # sensor_data = sensor_data[:2*rows//5, :] # thr front two fifths of all data
    # my_validator.plot_motions(sensor_data)
    # np.savetxt(os.path.join(my_validator.disp_folder, 'part_EQ1_S2PF1_data.txt'), sensor_data) # unit: mm
    
    sensor_data = np.loadtxt(os.path.join(my_validator.disp_folder, 'part_EQ1_S2PF1_data.txt')) # unit: mm
    
    # center_xys = my_validator.detect_video(video_name, start=0)
    # center_xys (time_stamps, center_xs, center_ys)
    center_xys = np.loadtxt(my_validator.detected_disp_folder+'/'+'time_center_xy_{0}.txt'.format(my_validator.experiment_name))
    video_start = 108
    sensor_start = 0
    disp_x = (center_xys[video_start:, :2] - center_xys[video_start, :2])
    disp_x[:, 1] = disp_x[:, 1]*scale*(-1)
    sensor_data = sensor_data[sensor_start:, :]  - sensor_data[sensor_start,:]
    my_validator.plot_motions(sensor_data, disp_x)
    
    np.savetxt(os.path.join(my_validator.detected_disp_folder, 'detected_disp_{0}.txt'.format(my_validator.experiment_name)), disp_x)
    
    # calculate the root of mean squared error
    rmse = my_validator.calculate_error(sensor_data, disp_x, mean=True)
    print(rmse)
    validation_data = np.loadtxt(os.path.join(my_validator.detected_disp_folder,
                                              'sampled_data_'+my_validator.experiment_name+'.txt'))
    plot_start = 200
    plot_end = 1100
    validation_data = validation_data[plot_start:plot_end+1]-validation_data[plot_start,:]
    rmse = np.sqrt(mean_squared_error(validation_data[:,1], validation_data[:,3]))
    my_validator.plot_motions(validation_data[:, :2], validation_data[:, 2:4])
    print('The final rmse:', rmse)
    print(validation_data[-1])
    np.savetxt(os.path.join(my_validator.detected_disp_folder,
                                              'validation_data_'+my_validator.experiment_name+'.txt'),
               validation_data)

def validate_PCW2(video_name='PRCW2_Drift1percent.mp4', scale=45/25):    # 1398/721*0.8
    """Validate the horizontal motion of the static cyclic PRCW2 shear wall experiment
        
        Args:
            video_name (str):           The name of the original recorded video
            scale (float):              The scale transfering units from pixel to mm
    """
    experiment_name = video_name.replace('.mp4', '')
    my_validator = Validator(experiment_name)

    sensor_file_name = 'disp_PRCW2_D22.txt' # time interval is 1 second
    sensor_data = np.loadtxt(os.path.join(my_validator.disp_folder, sensor_file_name)) # only displacement
    time_interval = 1.0
    time_length = len(sensor_data)*time_interval
    times = np.arange(0, time_length, time_interval)
    sensor_data = np.column_stack([times, sensor_data]) # (time, displacement)
    
    # crop = (0, 400, 220, 800) # (0, 400, 220, 800) # (100, 600, 200, 700) # (50, 500, 50+170, 500+300) # (100, 550, 200, 750)
    # center_xys = my_validator.detect_video(video_name, start=0, mask_box=crop)
    # center_xys (time_stamps, center_xs, center_ys)
    center_xys = np.loadtxt(my_validator.detected_disp_folder+'/'+'time_center_xy_{0}.txt'.format(my_validator.experiment_name))
    video_start = 0
    sensor_start = 0
    disp_y = (center_xys[video_start:, [0,2]] - center_xys[video_start, [0,2]])
    disp_y[:, 1] = disp_y[:, 1]*scale*(-1) # pixel to mm
    disp_x = (center_xys[video_start:, [0,1]] - center_xys[video_start, [0,1]])
    disp_x[:, 1] = disp_x[:, 1]*scale*(-1) # pixel to mm
    # high frequent data has a long lasting time
    # sensor data: low sampling frequency
    # detected data: high sampling frequency
    sensor_data = sensor_data[sensor_start:100, :]  - sensor_data[sensor_start, :]
    disp_y = disp_y[:3500, :]
    disp_x = disp_x[:3500, :]
    my_validator.plot_motions(sensor_data, disp_y, name='disp_y')
    my_validator.plot_motions(disp_x, name='disp_x')
    
    np.savetxt(os.path.join(my_validator.detected_disp_folder, 'detected_disp_{0}_y.txt'.format(my_validator.experiment_name)), disp_y)
    np.savetxt(os.path.join(my_validator.detected_disp_folder, 'detected_disp_{0}_x.txt'.format(my_validator.experiment_name)), disp_x)
    
    # calculate the root of mean squared error
    #                                   argument requirements:
    #                                   high frequency, low frequency
    #                                   long lasting time, short lasting time
    rmse_y = my_validator.calculate_error(disp_y, sensor_data, reverse=True, mean=True, direction='_y')
    rmse_x = my_validator.calculate_error(disp_x, sensor_data, reverse=True, mean=True, direction='_x')
    sampled_data_y = np.loadtxt(os.path.join(my_validator.detected_disp_folder,
                                              'sampled_data_'+my_validator.experiment_name+'_y.txt'))
    sampled_data_x = np.loadtxt(os.path.join(my_validator.detected_disp_folder,
                                              'sampled_data_'+my_validator.experiment_name+'_x.txt'))
    # sampled_data = np.sqrt(sampled_data_x[:,3]**2 + sampled_data_y[:,3]**2)
    # validation_data = np.hstack((sampled_data_y[:,:3], sampled_data_y.reshape(-1,1)))
    validation_data = sampled_data_y[10:31,:]
    validation_data = validation_data-validation_data[0,:]
    rmse = np.sqrt(mean_squared_error(validation_data[:,1], validation_data[:,3]))
    print('The final rmse:', rmse)
    my_validator.plot_motions(validation_data[:, :2], validation_data[:, 2:4], name='validation')
    print('The last data is', validation_data[-1])
    np.savetxt(os.path.join(my_validator.detected_disp_folder,
                                              'validation_data_'+my_validator.experiment_name+'.txt'),
               validation_data)

def compare():
    dynamic_test_data = np.loadtxt('./logs/detected_disp/validation_data_EQ1_DVR8_South.txt')
    static_test_data = np.loadtxt('./logs/detected_disp/validation_data_PRCW2_Drift1percent.txt')
    corr_dynamic = np.corrcoef(dynamic_test_data[:, 1], dynamic_test_data[:, 3])[0][1]
    corr_static = np.corrcoef(static_test_data[:, 1], static_test_data[:, 3])[0][1]
    # print(corr_dynamic, corr_static)
    from sklearn.metrics import r2_score, explained_variance_score
    r2_dynamic = r2_score(dynamic_test_data[:, 1], dynamic_test_data[:, 3])
    r2_static = r2_score(static_test_data[:, 1], static_test_data[:, 3])
    # print(r2_dynamic, r2_static)
    mape_dynamic = explained_variance_score(dynamic_test_data[:, 1], dynamic_test_data[:, 3])
    mape_static = explained_variance_score(static_test_data[:, 1], static_test_data[:, 3])
    print(mape_dynamic, mape_static)

if __name__ == '__main__':
    
    # validate_EQ1()
    # validate_PCW2()
    compare()
    
    # my_validator = Validator()
    # my_validator.get_fps()
    pass