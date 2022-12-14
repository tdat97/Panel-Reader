# from utils.camera import CameraManager
from utils.poly import SinglePolyDetector, get_crop_img_and_M
from utils.ocr import OcrEngine
from utils.db import DBManager
from utils.logger import logger, switch_logger_level
from utils.tools import *

import time
import argparse
import os
import cv2
import numpy as np
import re

# loop period (milli second)
LOOP_PERIOD = 30000

# Recode
RECODE_PATH = "./recode"
if not os.path.isdir(RECODE_PATH): os.mkdir(RECODE_PATH)
path = os.path.join(RECODE_PATH, "raw")
if not os.path.isdir(path): os.mkdir(path)
path = os.path.join(RECODE_PATH, "detect")
if not os.path.isdir(path): os.mkdir(path)
path = os.path.join(RECODE_PATH, "no_detect")
if not os.path.isdir(path): os.mkdir(path)

# Poly
SOURCE_IMG_PATH = "./source/image.png"
SOURCE_JSON_PATH = "./source/image_raw.json"
LABELS = ["panel", "TEMP_SV1", "TEMP_PV1", "RUN_ST"]
# LABELS = ["panel", "target_tmp", "actual_tmp", "run"]

# OCR
OCR_MODEL_PATH = "./source/date_ocr.h5"

# Color
NUM_PIXEL_BOUNDARY = 50 #400

# DB
TABLE_NAME = "tb_get_temp_f1p3"
TABLE_COLUMNS = ["TEMP_SV1", "TEMP_PV1", "RUN_ST"] # 설정값, 현재값, 가동

# Init
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
if not cam.isOpened(): logger.debug("Could not open webcam"); exit()
# cam warm up
status, img = cam.read()
status, img = cam.read()
status, img = cam.read()
logger.debug("cam opened.")

db_manager = DBManager()
logger.debug("db connected.")

poly_detector = SinglePolyDetector(SOURCE_IMG_PATH, SOURCE_JSON_PATH, pick_labels=LABELS)
logger.debug("poly_detector loaded.")

ocr_engine = OcrEngine(OCR_MODEL_PATH)
logger.debug("ocr_engine loaded.")


def main():
    print("To Exit, Press 'Ctrl+C' OR 'Q'")
    
    before_state = "1"
    show_img = None
    while True:
        # time.sleep(0.1)
        if show_img is not None: cv2.imshow("show", show_img)
        if cv2.waitKey(LOOP_PERIOD) & 0xff == ord('q'): break
        
        # file num manager
        manage_file_num(os.path.join(RECODE_PATH, "raw"))
        manage_file_num(os.path.join(RECODE_PATH, "detect"))
        manage_file_num(os.path.join(RECODE_PATH, "no_detect"))
        
        # shot and poly detection
        file_name = get_time_str() + ".jpg"
        status, img = cam.read()
        if not status: logger.warning(f"status : {status}"); continue
        polys, crop_imgs = poly_detector(img) # (panel, target_tmp, actual_tmp, run)
        path = os.path.join(RECODE_PATH, "raw", file_name)
        cv2.imwrite(path, img)
        
        # no detect
        if polys is None:
            logger.info("No Detected.")
            path = os.path.join(RECODE_PATH, "no_detect", file_name)
            cv2.imwrite(path, img)
            show_img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
            continue
        logger.info("Detected.")
        
        # ocr pred values
        values = [None] * len(LABELS)
        pick = [1,2]
        for i in pick:
            crop_img = crop_imgs[i]
            # crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
            pred_str = ocr_engine(crop_img).strip()
            values[i] = pred_str
            
        # check color
        pick = 3
        run_img_hsv = cv2.cvtColor(crop_imgs[pick], cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(run_img_hsv, (0, 50, 50), (60, 255, 250)) # red ~ yellow
        pixel_cnt = np.sum(color_mask//255)
        if NUM_PIXEL_BOUNDARY <= pixel_cnt: values[pick] = "ON"
        else: values[pick] = "OFF"
        logger.debug(f"pixel_cnt : {pixel_cnt}")
        
        # recode image
        img = draw_anno(img, LABELS, polys, values)
        path = os.path.join(RECODE_PATH, "detect", file_name)
        cv2.imwrite(path, img)
        show_img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
        logger.debug(f"values : {values}\t file_name : {file_name}")
        
        # fix values
        pick = [1,2]
        for i in pick:
            values[i] = re.sub(r'[^0-9]', '', values[i])
            values[i] = int(values[i]) if values[i] else 0
        pick = 3
        values[pick] = "1" if values[pick] == "ON" else "0"
        
        # transfer DB
        if before_state == values[3] == "0": continue
        value_dict = {}
        value_dict[TABLE_COLUMNS[0]] = values[1] # 설정값
        value_dict[TABLE_COLUMNS[1]] = values[2] # 현재값
        value_dict[TABLE_COLUMNS[2]] = values[3] # on/off
        db_manager.upload_data(TABLE_NAME, **value_dict)
        before_state = values[3]
            
    cam.release()
    cv2.destroyAllWindows()
    db_manager.close()
    
    
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglevel", type=str, default="DEBUG", 
                        help='["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_option()
    log_level = opt.__dict__["loglevel"]
    switch_logger_level(log_level)
    main()
