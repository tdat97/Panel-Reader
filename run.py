# from utils.camera import CameraManager
from utils.poly import SinglePolyDetector, get_crop_img_and_M
# from utils.ocr import OcrEngine
# from utils.transmit import TransmitDB
from utils.logger import logger, switch_logger_level
from utils.tools import *

import time
import argparse
import os
import cv2

# loop period (sec)
LOOP_PERIOD = 1

# Recode
RECODE_PATH = "./recode"
if not os.path.isdir(RECODE_PATH): os.mkdir(RECODE_PATH)
path = os.path.join(RECODE_PATH, "detect")
if not os.path.isdir(path): os.mkdir(path)
path = os.path.join(RECODE_PATH, "no_detect")
if not os.path.isdir(path): os.mkdir(path)

# Poly
SOURCE_IMG_PATH = "./source/test2.jpg"
SOURCE_JSON_PATH = "./source/test.json"
LABELS = ["panel", "target_tmp", "actual_tmp"]

# OCR
OCR_MODEL_PATH = "./source/date_ocr.h5"

# DB
# DB_ADDR = ""
# DB_USER = ""
# DB_PASS = ""

# Init
# cam_manager = CameraManager()
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
if not cam.isOpened(): print("Could not open webcam"); exit()
# cam warm up
status, img = cam.read()
status, img = cam.read()
status, img = cam.read()
logger.debug("cam opened.")

poly_detector = SinglePolyDetector(SOURCE_IMG_PATH, SOURCE_JSON_PATH, pick_labels=LABELS)
logger.debug("poly_detector loaded.")

ocr_engine = OcrEngine(OCR_MODEL_PATH)#, (2337, 100, 3))
logger.debug("ocr_engine loaded.")

# transmit_db = TransmitDB(DB_ADDR, DB_USER, DB_PASS)
# logger.debug("trainsmit_db loaded.")


def main(test_mode=False):
    logger.info(f"test mode : {test_mode}")
    print("To Exit, Press Ctrl+C")
    
    while True:
        time.sleep(LOOP_PERIOD)
        if cv2.waitKey(1) & 0xff == ord('q'): break
        
        # shot and poly detection
        file_name = get_time_str() + ".jpg"
        status, img = cam.read()
        if not status: logger.warning(f"status : {status}"); continue
        polys, crop_imgs = poly_detector(img)
        
        # no detect
        if polys is None:
            logger.info("no detect")
            path = os.path.join(RECODE_PATH, "no_detect", file_name)
            cv2.imwrite(path, img)
            if test_mode: cv2.imshow("test_show", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            continue
        
        # ocr pred values
        values = []
        for label, crop_img in zip(LABELS, crop_imgs):
            if label == LABELS[0]: values.append(""); continue
            crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
            pred_str = ocr_engine(crop_img).strip()
            values.append(pred_str)
        logger.info(f"values : {values}")

        
        # with poly_dict, Getting crop_img, pred_str
        # value_dict = {}
        # for label in ["target_tmp", "actual_tmp"]:
        #     poly = poly_dict[label]
        #     crop_img, _ = get_crop_img_and_M(img, poly)
        #     crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
        #     pred_str = ocr_engine(crop_img)
        #     value_dict[label] = pred_str.strip()
        # logger.info(f"value_dict : {value_dict}")
        
        
        # recode image
        img = draw_anno(img, LABELS, polys, values)
        path = os.path.join(RECODE_PATH, "detect", file_name)
        cv2.imwrite(path, img)
        
        # show or transmit
        if test_mode: cv2.imshow("test_show", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
        else:
            value_dict = {}
            for label, value in zip(LABELS[1:], values[1:]):
                if value.isdigit(): value_dict[label] = value
                else: value_dict[label] = ''
                
            # transmit_db(value_dict)
            
    cam.release()
    cv2.destroyAllWindows()
    
    
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument("--loglevel", type=str, default="DEBUG", 
                        help='["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_option()
    log_level = opt.__dict__["loglevel"]
    switch_logger_level(log_level)
    
    main(opt.__dict__["test"])
