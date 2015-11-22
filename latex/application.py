#!/usr/bin/env python
"""
Agri-Vision
Precision Agriculture and Soil Sensing Group (PASS)
McGill University, Department of Bioresource Engineering
"""

__author__ = 'Trevor Stanhope'
__version__ = '3.02'

## Libraries
import cv2, cv
import serial
import pymongo
from bson import json_util
from pymongo import MongoClient
import json
import numpy as np
from matplotlib import pyplot as plt
import thread
import gps
import time 
import sys
from datetime import datetime
import ast
import os
import threading
import cherrypy
from cherrypy.process.plugins import Monitor
from cherrypy import tools

## Constants
try:
    CONFIG_FILE = '%s' % sys.argv[1]
except Exception as err:
    cwd = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cwd, 'settings.cfg')
    CONFIG_FILE = open(config_path).read().rstrip()

# External Functions
def normalize_by_histogram(gray):
    """
    Normalize gray-scale by histogram
    """
    hist, bins = np.histogram(gray.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    gray_norm = cdf[gray] # Now we have the look-up table
    return gray_norm
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

## Class
class app:
    def __init__(self, config_file):

        # Path
        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load Config
        if config_file is None:
            self.default_settings()
        else:
            self.load_settings(config_file)
        
        # Initializers
        self.init_log() # it's best to run the log first to catch all events
        self.init_cameras()
        self.init_controller()
        self.init_pid()
        self.init_db()
        self.init_gps()
        self.init_display()
        self.init_webapp()

        # Thread states
        self.running = False
        self.updating = False

    # Initialize Webapp
    def init_webapp(self):
        self.log_msg('HTTP', 'Initializing Webapp Tasks')
        try:
            self.run_task = Monitor(cherrypy.engine, self.run, frequency=1/float(self.config['CAMERA_FPS'])).subscribe()
            if self.config['GPS_ENABLED']: self.gps_task = Monitor(cherrypy.engine, self.update_gps, frequency=1/float(self.config['GPS_HZ'])).subscribe()
            self.cameras_task = Monitor(cherrypy.engine, self.check_cameras, frequency=5).subscribe()
            self.controller_task = Monitor(cherrypy.engine, self.check_controller, frequency=5).subscribe()
            self.save_task =  Monitor(cherrypy.engine, self.save_image, frequency=1).subscribe()
        except Exception as error:
            self.log_msg('ENGINE', 'Error: %s' % str(error), important=True)

    # Initialize Camera(s)
    def init_cameras(self):
        
        # Setting variables
        self.log_msg('CAM', 'Initializing CV Variables')
        self.log_msg('CAM', 'Camera Width: %d px' % self.config['CAMERA_WIDTH'])
        self.log_msg('CAM', 'Camera Height: %d px' % self.config['CAMERA_HEIGHT'])
        self.log_msg('CAM', 'Camera Depth: %d cm' % self.config['CAMERA_DEPTH'])
        self.log_msg('CAM', 'Camera FOV: %f cm' % self.config['CAMERA_FOV'])
        self.log_msg('CAM', 'Error Tolerance: +/- %1.1f cm' % self.config['ERROR_TOLERANCE']) 
        
        # Attempt to set each camera index/name
        self.log_msg('CAM', 'Initializing Cameras')
        self.cameras = []
        self.images = []
        for i in range(self.config['CAMERAS']):
            try:
                cam = self.attach_camera(i) # returns None if camera failed to attach
                self.cameras.append(cam)
                self.images.append(np.zeros((self.config['CAMERA_HEIGHT'], self.config['CAMERA_WIDTH'], 3), np.uint8))
                self.log_msg('CAM', 'OK: Camera #%d connected' % i)
            except Exception as error:
                self.log_msg('CAM', 'ERROR: %s' % str(error), important=True)

    ## Attach a camera capture
    def attach_camera(self, i):
        try:
            self.log_msg('CAM', 'WARN: Attaching Camera #%d' % i)
            cam = cv2.VideoCapture(i)
            cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.config['CAMERA_WIDTH'])
            cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.config['CAMERA_HEIGHT'])
            cam.set(cv.CV_CAP_PROP_SATURATION, self.config['CAMERA_SATURATION'])
            cam.set(cv.CV_CAP_PROP_BRIGHTNESS, self.config['CAMERA_BRIGHTNESS'])
            cam.set(cv.CV_CAP_PROP_CONTRAST, self.config['CAMERA_CONTRAST'])
            cam.set(cv.CV_CAP_PROP_FPS, self.config['CAMERA_FPS'])
            if cam.isOpened():
                self.log_msg('CAM', 'OK: Camera successfully opened')
                return cam
            else:
                self.log_msg('CAM', 'ERROR: Failed to attach camera!')
                cam.release()
                return None
        except:
            self.log_msg("CAM", "Failed to create camera object!", important=True)
            return None

    # Initialize Database
    def init_db(self):
        self.LOG_NAME = datetime.strftime(datetime.now(), self.config['LOG_FORMAT'])
        self.MONGO_NAME = datetime.strftime(datetime.now(), self.config['MONGO_FORMAT'])
        self.log_msg('DB', 'Initializing MongoDB')
        self.log_msg('DB', 'Connecting to MongoDB: %s' % self.MONGO_NAME)
        self.log_msg('DB', 'New session: %s' % self.LOG_NAME)
        try:
            self.client = MongoClient()
            self.database = self.client[self.MONGO_NAME]
            self.collection = self.database[self.LOG_NAME]
            self.log_msg('DB', 'Setup OK')
        except Exception as error:
            self.log_msg('DB', 'ERROR: %s' % str(error), important=True)
    
    # Initialize PID Controller
    def init_pid(self):
        self.log_msg('PID', 'Initialing Electro-Hydraulics')
        try:
            self.log_msg('PID', 'N Samples: : %s' % self.config['N_SAMPLES'])
            self.log_msg('PID', 'P Coefficient: : %s' % self.config['P_COEF'])
            self.log_msg('PID', 'I Coefficient: : %s' % self.config['I_COEF'])
            self.log_msg('PID', 'D Coefficient: : %s' % self.config['D_COEF'])
            self.offset_history = [ 0.0 ] * int(self.config['N_SAMPLES'])
            self.log_msg('PID', 'Setup OK')
        except Exception as error:
            self.log_msg('PID', 'ERROR: %s' % str(error), important=True)
    
    # Initialize Log
    def init_log(self):
        try:
            self.LOG_NAME = datetime.strftime(datetime.now(), self.config['LOG_FORMAT'])
            error_log_path = os.path.join(self.CURRENT_DIR, self.config['LOG_DIR'], self.config['LOG_MESSAGES'])
            self.error_log = open(error_log_path, 'w')
            self.log_msg('LOG', 'Message Setup OK')
            if self.config['DATALOG_ON']:
                data_log_path = os.path.join(self.CURRENT_DIR, self.config['LOG_DIR'], self.LOG_NAME)
                self.data_log = open(data_log_path, 'w')
                headers = ['time', 'estimated', 'sig', 'pwm', 'volts']
                self.data_log.write(','.join(headers) + '\n')
            self.log_msg('LOG', 'Datalog Setup OK')
        except Exception as error:
            self.log_msg('ERROR', str(error), important=True)
            
    # Initialize Controller
    def init_controller(self):
        self.log_msg('CTRL', 'Initializing controller ...')
        self.calibrating = False
        try:
            self.log_msg('CTRL', 'Device: %s' % self.config['SERIAL_DEVICE'])
            self.log_msg('CTRL', 'Baud Rate: %d' % self.config['SERIAL_BAUD'])
            self.log_msg('CTRL', 'PWM Resolution: %d' % self.config['PWM_RESOLUTION'])
            self.log_msg('CTRL', 'Voltage: %d' % self.config['PWM_VOLTAGE'])
            self.controller = self.attach_controller()
            if self.controller is None:
                self.log_msg('CTRL', 'ERROR: Could not locate a controller!')
            else:
                self.log_msg('CTRL', 'Setup Succesful')
        except Exception as error:
            self.log_msg('CTRL', 'ERROR: %s' % str(error), important=True)
            self.controller = None
    
    ## Attach Controller
    def attach_controller(self, attempts=5):
        try:
            self.log_msg('CTRL', 'Attaching controller ...')
            for dev in self.config['SERIAL_DEVICE']:
                for i in range(attempts):
                    dev_num = dev + str(i)
                    self.log_msg('CTRL', 'Trying: %s' % dev_num)
                    try:
                        ctrl = serial.Serial(dev_num, self.config['SERIAL_BAUD'])
                        return ctrl
                    except Exception as err:
                        self.log_msg('CTRL', 'WARN: %s' % str(err), important=True)
            return None
        except Exception as error:
            raise Exception("Serial connection failed!", important=True)

    # Initialize GPS
    def init_gps(self):
        self.log_msg('GPS', 'Initializing GPS ...')
        self.latitude = 0
        self.longitude = 0
        self.speed = 0
        try:
            self.log_msg('GPS', 'Enabing GPS ...')
            self.gpsd = gps.gps()
            self.gpsd.stream(gps.WATCH_ENABLE)
        except Exception as err:
            self.gpsd = None
            self.log_msg('GPS', 'WARNING: GPS not available! %s' % str(err), important=True)
    
    # Display
    def init_display(self):
        self.log_msg('DISP', 'Initializing Display')
        try:
            self.output_image = np.zeros((self.config['CAMERA_HEIGHT'], self.config['CAMERA_WIDTH'], 3), np.uint8)
            if not self.config['DISPLAY_ON']:
                self.log_msg('DISP', 'WARN: Display disabled!')
        except Exception as error:
            self.log_msg('DISP', 'ERROR: %s' % str(error), important=True)

    ## Rotate image
    def rotate_image(self, bgr):
        bgr = cv2.rotate(bgr)
        return bgr
    
    ## Check Cameras
    def check_cameras(self):
        """ Checks cameras for functioning streams """
        if self.config['VERBOSE']: self.log_msg('CAM', 'Checking Cameras ...')
        for i in range(self.config['CAMERAS']):
            if self.cameras[i] is None:
                self.cameras[i] = self.attach_camera(i)
            elif not self.cameras[i].isOpened():
                self.log_msg('CAM', 'Camera #%d is offline' % i)
                self.cameras[i] = None
            else:
                self.log_msg('CAM', 'Camera #%d is okay' % i)

    ## Capture images from multiple cameras #!TODO: fix halting
    def capture_images(self):
        a = time.time()
        if self.config['VERBOSE']: self.log_msg('CAM', 'Capturing Images ...')
        images = [None] * self.config['CAMERAS']
        for i in range(self.config['CAMERAS']):
            self.log_msg('CAM', 'Attempting on Camera #%d' % i)
            try:
                cam = self.cameras[i]
                if cam is not None:
                    (s, bgr) = self.cameras[i].read()
                    if s:
                        if np.all(bgr==self.images[i]): # Check to see if frame is frozen
                            self.log_msg('CAM', 'WARN: Frozen frame')
                            self.cameras[i].release()
                        else:
                            self.log_msg('CAM', 'Capture successful: %s' % str(bgr.shape))
                            images[i] = bgr
                    else:
                        self.log_msg('CAM', 'WARN: Capture failed')
                        self.cameras[i].release()
                else:
                    pass
            except:
                pass
        if all(v is None for v in images): raise Exception("No images captured!", important=True)
        self.images = images
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('CAM', '... %.2f ms' % ((b - a) * 1000))
        return images

        
    ## Plant Segmentation Filter
    def bppd_filter(self, images):    
        """
        1. RBG --> HSV
        2. Set minimum saturation equal to the mean saturation
        3. Set minimum value equal to the mean value
        4. Take hues within range from green-yellow to green-blue
        """
        if self.config['VERBOSE']: self.log_msg('BPPD', 'Filtering for plants ...')
        if images == []: raise Exception("No input image(s)!", important=True)
        a = time.time()
        masks = []
        threshold_min = np.array([self.config['HUE_MIN'], self.config['SAT_MIN'], self.config['VAL_MIN']], np.uint8)
        threshold_max = np.array([self.config['HUE_MAX'], self.config['SAT_MAX'], self.config['VAL_MAX']], np.uint8)
        for bgr in images:
            if bgr is not None:
                try:
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    threshold_min[1] = np.percentile(hsv[:,:,1], 100 * self.config['SAT_MIN'] / 255.0)
                    threshold_min[2] = np.percentile(hsv[:,:,2], 100 * self.config['VAL_MIN'] / 255.0)
                    threshold_max[1] = np.percentile(hsv[:,:,1], 100 * self.config['SAT_MAX'] / 255.0)
                    threshold_max[2] = np.percentile(hsv[:,:,2], 100 * self.config['VAL_MAX'] / 255.0)
                    mask = cv2.inRange(hsv, threshold_min, threshold_max)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((self.config['KERNEL_X'],self.config['KERNEL_Y']), np.uint8))
                    masks.append(mask)
                    self.log_msg('BPPD', 'Mask #%d was successful' % len(masks))                    
                except Exception as error:
                    self.log_msg('BPPD', str(error), important=True)
            else:
                self.log_msg('BPPD', 'WARN: Mask #%d is blank' % len(masks), important=True)
                masks.append(None)
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('BPPD', '... %.2f ms' % ((b - a) * 1000))
        return masks
        
    ## Find Plants
    def find_offset(self, masks):
        """
        1. Calculates the column summation of the mask
        2. Calculates the 95th percentile threshold of the column sum array
        3. Finds indicies which are greater than or equal to the threshold
        4. Finds the median of this array of indices
        5. Repeat for each mask
        """
        if self.config['VERBOSE']: self.log_msg('BPPD', 'Finding offset of row ...')
        if masks == []: raise Exception("No input mask(s)!", important=True)
        a = time.time()
        offsets = []
        sums = []
        for mask in masks:
            if mask is not None:
                try:
                    colsum = mask.sum(axis=0) # vertical summation
                    T = np.percentile(colsum, self.config['THRESHOLD_PERCENTILE'])
                    probable = np.nonzero(colsum >= T) # returns 1 length tuble
                    if self.config['DEBUG']: pass
                    num_probable = len(probable[0])
                    best = int(np.median(probable[0]))
                    s = colsum[best]
                    c = best - self.config['CAMERA_WIDTH'] / 2
                    offsets.append(c)
                    sums.append(s)
                except Exception as error:
                    self.log_msg('OFF', '%s' % str(error), important=True)
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('OFF', '... %.2f ms' % ((b - a) * 1000))
        return offsets, sums
        
    ## Best Guess for row based on multiple offsets from indices
    def estimate_row(self, indices, sums, tol=32):
        """
        1. If outside bounds, ignore
        2. If inside, check difference between indices from both cameras
        3. If similar, take mean, else take dominant estimation
        1. Takes the current assumed offset and number of averages
        2. Calculate weights of previous offset
        3. Estimate the weighted position of the crop row (in pixels)
        """
        if self.config['VERBOSE']:self.log_msg('ROW', 'Smoothing offset estimation ...')
        if sums == []: raise Exception("No input sum(s)!", important=True)
        if indices == []: raise Exception("No input indice(s)", important=True)
        if len(sums) != len(indices): raise Exception("len(x) must equal len(y)", important=True)
        a = time.time()
        try:
            e = np.abs(np.mean(np.diff(indices)))
            sums = np.array(sums)
            if e > self.config['ERROR_TOLERANCE'] * self.config['CAMERA_WIDTH'] / self.config['CAMERA_FOV']:
                indices = np.array(indices)
                cur =  indices[np.argmax(sums)]
            else:
                cur = np.mean(indices)
        except Exception as error:
            self.log_msg('ROW', 'ERROR: %s' % str(error), important=True)
            est = self.config['CAMERA_WIDTH'] / 2

        # Insert current into local memory
        self.offset_history.append(cur)
        lim = int(np.ceil(self.config['N_SAMPLES'] / float(self.config['AGGRESSIVENESS'])))
        while len(self.offset_history) > lim:
            self.offset_history.pop(0)

        # Smooth
        n = int(np.ceil(self.config['N_SAMPLES'] / float(self.config['AGGRESSIVENESS'])))
        est =  int(self.offset_history[-1]) #!TODO
        avg = int(np.mean(self.offset_history[-n:])) #!TODO
        diff = int(np.mean(np.diff(self.offset_history[-n:]))) #!TODO can be a little more clever e.g. np.gradient, np.convolve
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('ROW', '... %.2f ms' % ((b - a) * 1000))
        return est, avg, diff
         
    ## Control Hydraulics
    def calculate_output(self, est, avg, diff):
        """
        Calculates the PID output for the PWM controller
        Arguments: est, avg, diff
        Requires: PWM_VOLTAGE, PWM_RESOLUTION, MAX_VOLTAGE, MIN_VOLTAGE
        Returns: PWM
        """
        if self.config['VERBOSE']: self.log_msg('PID', 'Calculating PID Output ...')
        a = time.time()
        try:
            p = est * self.config['P_COEF']
            i = avg * self.config['I_COEF']
            d = diff  * self.config['D_COEF']
            sig = self.config['SENSITIVITY'] * (p + i + d) - self.config['CAMERA_OFFSET'] # scale and offset signal
            self.log_msg('PID', "sig = %.1f + %.1f + %.1f" % (p,i,d))         
        except Exception as error:
            self.log_msg('PID', 'ERROR: %s' % str(error))
            sig = 0
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('PID', '... %.2f ms' % ((b - a) * 1000))
        return sig
    
    ## To Volts
    def to_volts(self, pwm):
        volts_at_ctrl = (pwm * self.config['PWM_VOLTAGE']) / float(self.config['PWM_RESOLUTION'])
        volts = round(np.interp(volts_at_ctrl, [0, self.config['PWM_VOLTAGE']], [0, self.config['SUPPLY_VOLTAGE']]) / 1000.0, 2)  
        self.log_msg('PID', 'PWM = %d (%.2f V)' % (pwm, volts))
        return volts

    ## Control Hydraulics
    def set_controller(self, sig):
        """
        1. Get PWM response corresponding to average offset
        2. Send PWM response over serial to controller
        """
        a = time.time()
        if self.config['VERBOSE']: self.log_msg('CTRL', 'Setting controller state ...')
        try:            
            if self.controller is None: raise Exception("No controller!")
            v_zero = (self.config['MAX_VOLTAGE'] + self.config['MIN_VOLTAGE']) / 2.0
            pwm_zero = np.interp(v_zero, [0, self.config['SUPPLY_VOLTAGE']], [0, self.config['PWM_RESOLUTION']])
            pwm_max = np.interp(self.config['MAX_VOLTAGE'], [0, self.config['SUPPLY_VOLTAGE']], [1, self.config['PWM_RESOLUTION']])
            pwm_min = np.interp(self.config['MIN_VOLTAGE'], [0, self.config['SUPPLY_VOLTAGE']], [1, self.config['PWM_RESOLUTION']])
            if self.calibrating == True:
                return pwm_zero
            else:
                pwm = sig + pwm_zero # offset to zero
                if pwm > pwm_max:
                    pwm = pwm_max
                elif pwm < pwm_min:
                    pwm = pwm_min
                if self.config['PWM_INVERTED']:
                    pwm = pwm_max - pwm + pwm_min
                self.controller.write(str(int(pwm)) + '\n') # Write to PWM adaptor
                res = self.controller.readline()
                if int(res) != int(pwm):
                    self.log_msg('CTRL', 'WARN: Controller returned bad value!', important=True)
                else:
                    self.log_msg('CTRL', 'OK: Set controller successfully')
                return pwm
        except Exception as error:
            self.log_msg('CTRL', 'ERROR: %s' % str(error), important=True)
            #!TODO add bit here to automatically kill controller object now?
            try:
                self.controller.close()
            except:
                pass
            self.controller = None
        b = time.time()
        if self.config['VERBOSE']: self.log_msg('CTRL', '... %.2f ms' % ((b - a) * 1000))
    
    ## Log to Mongo
    def log_db(self, sample):
        """
        1. Log results to the database
        2. Returns Doc ID
        """
        if self.config['VERBOSE']: self.log_msg('MDB', 'Writing to database ...')
        try:          
            if self.collection is None: raise Exception("Missing DB collection!")
            doc_id = self.collection.insert(sample)
            self.log_msg('MDB', 'Doc ID: %s' % str(doc_id))
        except Exception as error:
            self.log_msg('MDB', 'ERROR: %s' % str(error), important=True)
        return doc_id
    
    ## Log to File
    def log_data(self, data):
        """
        1. Open new text file
        2. For each document in session, print parameters to file
        """
        try:
            if self.data_log is None: raise Exception("Missing data logfile!")
            t = datetime.strftime(datetime.now(), self.config['TIME_FORMAT'])
            data_as_str = [str(d) for d in data]
            self.data_log.write(','.join([t] + data_as_str + ['\n']))
        except Exception as error:
            self.log_msg('LOG', 'ERROR: %s' % str(error), important=True)
    
    ## Log Messages
    def log_msg(self, header, msg, important=False):
        """
        Saves important messages to logfile
        """
        try:
            if self.error_log is None: raise Exception("Missing error logfile!")
            date = datetime.strftime(datetime.now(), self.config['TIME_FORMAT'])
            formatted_msg = "%s\t%s\t%s" % (date, header, msg)
            self.error_log.write(formatted_msg + '\n')
            if self.config['VERBOSE'] or important: print formatted_msg
        except Exception as error:
            print "%s\tLOG\tERROR: Failed to log message!\n" % date
    
    ## Display window briefly
    def flash_window(self, img, title=''):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if self.config['FULLSCREEN']: cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.imshow(title, img)
        if cv2.waitKey(5) == 0:
            time.sleep(0.05)            
            pass

    ## Draw Lines
    def draw_lines(self, img, offset):
        """
        Draw tolerance, offset, and center lines on image
        """
        (h, w, d) = img.shape
        try:
            TOLERANCE = int((self.config['CAMERA_WIDTH'] / self.config['CAMERA_FOV']) * self.config['ERROR_TOLERANCE']) 
            cv2.line(img, (TOLERANCE + w/2 + self.config['CAMERA_OFFSET'], 0), (TOLERANCE + w/2 + self.config['CAMERA_OFFSET'], h), (0,0,255), 1)
            cv2.line(img, (-TOLERANCE + w/2 + self.config['CAMERA_OFFSET'], 0), (-TOLERANCE + w/2 + self.config['CAMERA_OFFSET'], h), (0,0,255), 1)   
            cv2.line(img, (offset + w/2, 0), (offset + w/2, h), (0,255,0), 2) # Average
            cv2.line(img, (w/2 + self.config['CAMERA_OFFSET'], 0), (w/2 + self.config['CAMERA_OFFSET'], h), (255,255,255), 1) # Center
            return img
        except Exception as e:
            raise e

    ## Run Calibration
    def run_calibration(self, delay=1.0, sweeps=2):
        self.calibrating = True
        self.log_msg('LOG', 'WARN: Running calibration sweep ...', important=True)
        v_zero = (self.config['MAX_VOLTAGE'] + self.config['MIN_VOLTAGE']) / 2.0
        pwm_zero = np.interp(v_zero, [0, self.config['SUPPLY_VOLTAGE']], [0, self.config['PWM_RESOLUTION']])
        pwm_max = np.interp(self.config['MAX_VOLTAGE'], [0, self.config['SUPPLY_VOLTAGE']], [1, self.config['PWM_RESOLUTION']])
        pwm_min = np.interp(self.config['MIN_VOLTAGE'], [0, self.config['SUPPLY_VOLTAGE']], [1, self.config['PWM_RESOLUTION']])
        try: 
            for i in range(sweeps):
                # Hold at min
                self.controller.write(str(int(pwm_min)) + '\n') # Write to PWM adaptor
                self.controller.readline()
                time.sleep(delay)
                # Sweep up     
                for pwm in range(int(pwm_min), int(pwm_max)):
                    self.controller.write(str(int(pwm)) + '\n') # Write to PWM adaptor
                    self.controller.readline()
                # Hold at max
                self.controller.write(str(int(pwm_max)) + '\n') # Write to PWM adaptor
                self.controller.readline()
                time.sleep(delay)
                # Sweep down            
                for pwm in range(int(pwm_max), int(pwm_min)):
                    self.controller.write(str(int(pwm)) + '\n') # Write to PWM adaptor
                    self.controller.readline()
        except Exception as error:
            self.log_msg('LOG', 'ERROR: %s' % str(error), important=True)
        self.calibrating = False

    ## Generate Output Image
    def generate_output(self, images, masks, avg, volts):
        """
        Generates a single composite image of the multiple camera feeds, with lines drawn
        """
        output_images = []
        if self.config['HIGHLIGHT']:
            if self.config['VERBOSE']: self.log_msg('DISP', 'Using mask for output images')
            for mask in masks:
                try:
                    if mask is None: mask = np.zeros((self.config['CAMERA_HEIGHT'], self.config['CAMERA_WIDTH'], 1), np.uint8)
                    img = np.array(np.dstack((mask, mask, mask)))
                    output_images.append(self.draw_lines(img, avg))
                except Exception as error:
                    raise error
        else:
            if self.config['VERBOSE']: self.log_msg('DISP', 'Using RGB for output images')
            for img in images:
                try:
                    if img is None: img = np.zeros((self.config['CAMERA_HEIGHT'], self.config['CAMERA_WIDTH'], 3), np.uint8)
                    output_images.append(self.draw_lines(img, avg))
                except Exception as error:
                    raise error
        output = np.vstack(output_images)

        # Add Padding
        pad_y = self.config['CAMERA_HEIGHT'] * 0.15
        pad = np.zeros((pad_y, self.config['CAMERA_WIDTH'], 3), np.uint8) # add blank space
        output = np.vstack([output, pad])

        # Offset Distance
        distance = round((avg - self.config['CAMERA_OFFSET']) / (self.config['CAMERA_WIDTH'] / self.config['CAMERA_FOV']), 2)
        self.log_msg('DISP', 'Offset Distance: %d' % distance)
        if avg - self.config['CAMERA_WIDTH'] / 2 >= 0:
            distance_str = str("+%2.1f cm" % distance)
        elif avg - self.config['CAMERA_WIDTH'] / 2 < 0:
            distance_str = str("%2.1f cm" % distance)
        else:
            distance_str = str("  .  cm")
        cv2.putText(output, distance_str, (int(self.config['CAMERA_WIDTH'] * 0.04), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                
        # Output Voltage
        volts_str = str("%2.1f V" % volts)
        cv2.putText(output, volts_str, (int(self.config['CAMERA_WIDTH'] * 0.72), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                
        # Arrow (Directional)
        if avg - self.config['CAMERA_WIDTH'] / 2 >= 0:
            p = (int(self.config['CAMERA_WIDTH'] * 0.45), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 2))
            q = (int(self.config['CAMERA_WIDTH'] * 0.55), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 2))
        elif avg - self.config['CAMERA_WIDTH'] / 2 < 0:
            p = (int(self.config['CAMERA_WIDTH'] * 0.55), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 2))
            q = (int(self.config['CAMERA_WIDTH'] * 0.45), int(self.config['CAMERAS'] * self.config['CAMERA_HEIGHT'] + pad_y / 2))
        color = (255,255,255)
        thickness = 4
        line_type = 8
        shift = 0
        arrow_magnitude=15
        cv2.line(output, p, q, color, thickness, line_type, shift) # draw arrow tail
        angle = np.arctan2(p[1]-q[1], p[0]-q[0])
        p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)), # starting point of first line of arrow head 
        int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
        cv2.line(output, p, q, color, thickness, line_type, shift) # draw first half of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)), # starting point of second line of arrow head 
        int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
        cv2.line(output, p, q, color, thickness, line_type, shift) # draw second half of arrow head
        return output

    ## Update the Display
    def update_display(self, images, masks, volts, avg):
        """
        0. Check for concurrent update process
        1. Draw lines on RGB images
        2. Draw lines on ABP masks
        3. Output GUI display
        """
        if self.config['VERBOSE']: self.log_msg('DISP', 'Updating display...')
        a = time.time()
        if self.updating:
            return # if the display is already updating, wait and exit (puts less harm on the CPU)
        else:
            self.updating = True
            try:
                output_small = self.generate_output(images, masks, avg, volts)
                self.output_image = output_small
                output_large = cv2.resize(output_small, (self.config['DISPLAY_WIDTH'], self.config['DISPLAY_HEIGHT']))

                # Draw GUI
                self.log_msg('DISP', 'Output shape: %s' % str(output_large.shape))
                self.flash_window(output_large, title='Agri-Vision')
            except Exception as error:
                self.updating = False
                self.log_msg('DISP', str(error), important=True)
            self.updating = False
        b = time.time()
        self.log_msg('DISP', '... %.2f ms' % ((b - a) * 1000))
                    
    ## Update GPS
    def update_gps(self):
        """
        1. Get the most recent GPS data
        2. Set global variables for lat, long and speed
        """
        if self.config['GPS_ENABLED']:
            if self.gpsd is not None:
                try:
                    self.gpsd.next()
                    self.latitude = self.gpsd.fix.latitude
                    self.longitude = self.gpsd.fix.longitude
                    self.speed = self.gpsd.fix.speed
                    self.log_msg('GPS', '%d N %d E' % (self.latitude, self.longitude))
                except Exception as error:
                    self.log_msg('GPS', 'ERROR: %s' % str(error), important=True)
    
    ## Close
    def close(self, delay=0.5):
        """
        Function to shutdown application safely
        1. Close windows
        2. Disable controller
        3. Release capture interfaces 
        """
        self.log_msg('SYS', 'Shutting Down!')
        if self.controller is None:
            self.log_msg('WARN', 'Controller already off!')
        else:
            try:
                self.log_msg('CTRL', 'Closing Controller ...')
                self.controller.close() ## Disable controller
            except Exception as error:
                self.log_msg('CTRL', 'ERROR: %s' % str(error), important=True)
        for i in range(len(self.cameras)):
            if self.cameras[i] is None:
                self.log_msg('CAM', 'WARN: Camera %d already off!' % i)
            else:
                try:
                    self.log_msg('CAM', 'WARN: Closing Camera %d ...' % i)
                    self.cameras[i].release() ## Disable cameras
                except Exception as error:
                    self.log_msg('CAM', 'ERROR: %s' % str(error), important=True)
        if self.config['DISPLAY_ON']:
            cv2.destroyAllWindows() ## Close windows
  
    ## Check Controller
    def check_controller(self):
        """ Checks controller for functioning stream """
        if self.config['VERBOSE']: self.log_msg('CTRL', 'Checking controller ...')
        if self.controller is None:
            try:
                self.controller = self.attach_controller()
            except Exception as e:
                self.log_msg('CTRL', 'ERROR: Failed to attach controller!', important=True)

    ## Run  
    def run(self):
        """
        Function for Run-time loop
        1. Get initial time
        2. Capture images
        3. Generate mask filter for plant matter
        4. Calculate indices of rows
        5. Estimate row from both images
        6. Get number of averages
        7. Calculate moving average
        8. Send PWM response to controller
        9. Throttle to desired frequency
        10. Log results to DB
        11. Display results
        """
        if self.running: 
            return # Don't do anything
        else:
            self.running = True
            try:
                images = self.capture_images()
                masks = self.bppd_filter(images)
                offsets, sums = self.find_offset(masks)
                (est, avg, diff) = self.estimate_row(offsets, sums)
                sig = self.calculate_output(est, avg, diff)
                pwm = self.set_controller(sig)
                volts = self.to_volts(pwm)
                self.log_msg("SYS", "err:%s, cams:%d, pwm:%d, volts:%2.2f" % (str(offsets), avg, pwm, volts), important=True)
                if self.config['MONGO_ON']: doc_id = self.log_db(sample)
                if self.config['DATALOG_ON']: self.log_data([est, sig, pwm, volts])
                if self.config['DISPLAY_ON']:
                    try:
                        os.environ['DISPLAY']
                        threading.Thread(target=self.update_display,
                            args=(images, masks, volts, avg),
                            kwargs={}
                        ).start()
                    except Exception as error:
                        self.log_msg('SYS', 'ERROR: %s' % str(error), important=True)
                else:
                    self.output_image = self.generate_output(images, masks, avg, volts)  
            except KeyboardInterrupt as error:
                self.shutdown()
            except UnboundLocalError as error:
                self.log_msg('SYS', 'ERROR: %s' % str(error), important=True)
                self.shutdown()
            except IOError as error:
                self.log_msg('SYS', 'ERROR: %s' % str(error), important=True)
                self.shutdown()
            except Exception as error:
                self.log_msg('SYS', 'ERROR: %s' % str(error))
                try:
                    for i in range(self.config['CAMERAS']):
                        self.attach_camera(i)
                except Exception as error:
                    self.log_msg('SYS', 'ERROR: %s' % str(error), important=True)
            self.running = False

    ## Save Image
    def save_image(self, filename='out.jpg', subdir='agionic/www'):
        """
        Save image to output file (to be loaded by webserver)
        """
        try:
            if self.config['VERBOSE']: self.log_msg('HTTP', 'Saving output image to file')
            filepath = os.path.join(self.CURRENT_DIR, subdir, filename)
            cv2.imwrite(filepath, self.output_image)
        except Exception as error:
            self.log_msg('SYS', 'ERROR: %s' % str(error), important=True)

    ## Save Settings
    def save_settings(self, keyval, filename='custom.json', subdir='modes'):
        """
        Save current config to a custom config file
        """
        keys = keyval.keys()
        vals = [int(v) for v in keyval.values()]
        new_settings = dict(zip(keys, vals))
        self.log_msg('CONF', str(new_settings))
        self.config.update(new_settings)
        filepath = os.path.join(self.CURRENT_DIR, subdir, filename)
        with open(filepath, 'w') as jsonfile:
            jsonfile.write(json.dumps(self.config, indent=4, sort_keys=True))
    
    ## Default Settings
    def default_settings(self, filename='default.json', subdir='modes'):
        """
        Resets config to default settings file
        """
        filepath = os.path.join(self.CURRENT_DIR, subdir, filename)
        print "WARNING: Loading %s" % filepath
        if os.path.exists(filepath):
            with open(filepath, 'r') as jsonfile:
                self.config = json.loads(jsonfile.read())
        else:
            print "FATAL ERROR: No config found!"
            exit(1)

    ## Load Settings
    def load_settings(self, config_file, subdir='modes'):
        """
        Load config from input file
        """
        filepath = os.path.join(self.CURRENT_DIR, subdir, config_file)
        print "WARNING: Loading %s" % filepath
        if os.path.exists(filepath):
            with open(filepath, 'r') as jsonfile:
                self.config = json.loads(jsonfile.read())
        else:
            print "ERROR: Specified config not found! Trying default!"            
            self.default_settings()

    ## Render Index
    @cherrypy.expose
    def index(self, indexfile="index.html"):
        indexpath = os.path.join(self.CURRENT_DIR, self.config['CHERRYPY_PATH'], indexfile)
        with open(indexpath) as html:
            return html.read()

    ## Handle Posts
    @cherrypy.expose
    def default(self, *args, **kwargs):
        """
        This function is basically the API
        """
        try:
            url = args[0] #!TODO: select action depending on request type/url
            #self.save_image()
            if url == 'update':       
                # Parse JSON object to pythonic dict 
                post = kwargs.keys()
                data = post[0]
                unicode_config = json.loads(data)
                new_configs = dict([(str(k), v) for k, v in unicode_config.items()])                
                self.save_settings(new_configs)
            elif url == 'config':
                self.log_msg("HTTP", "Caught /config request")
                return json.dumps(self.config)
            elif url == 'default':
                self.log_msg("HTTP", "Caught /default request")
                self.default_settings()
                return json.dumps(self.config)
            elif url == 'calibrate':
                self.log_msg("HTTP", "Caught /calibrate request")
                self.run_calibration()
        except Exception as err:
            self.log_msg('ERROR', str(err), important=True)
        return None

    ## CherryPy Reboot
    @cherrypy.expose
    def shutdown(self):
        cherrypy.engine.exit()

## Main
if __name__ == '__main__':
    session = app(CONFIG_FILE)
    cherrypy.server.socket_host = session.config['CHERRYPY_ADDR']
    cherrypy.server.socket_port = session.config['CHERRYPY_PORT']
    conf = {
        '/': {'tools.staticdir.on':True, 'tools.staticdir.dir':os.path.join(session.CURRENT_DIR, session.config['CHERRYPY_PATH'])},
        '/js': {'tools.staticdir.on':True, 'tools.staticdir.dir':os.path.join(session.CURRENT_DIR, session.config['CHERRYPY_PATH'], 'js')},
        '/logs': {'tools.staticdir.on':True, 'tools.staticdir.dir':os.path.join(session.CURRENT_DIR, 'logs')},
    }
    cherrypy.quickstart(session, '/', config=conf)
    session.close()
    #if session.reboot: os.system("reboot")
