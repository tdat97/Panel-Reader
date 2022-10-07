from utils.camera import CameraManager
from utils.poly import SinglePolyDetector, get_crop_img_and_M
from utils.ocr import OcrEngine
# from utils.transmit import TransmitDB
from utils.logger import logger, switch_logger_level
from utils.tools import *

import time
import argparse
import os

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
SOURCE_IMG_PATH = "./source/panel2.png"
SOURCE_JSON_PATH = "./source/panel.json"
TARGET_LABEL = "panel"

# OCR
OCR_MODEL_PATH = "./source/OCR_aug3_300k.h5"

# DB
# DB_ADDR = ""
# DB_USER = ""
# DB_PASS = ""

# Init
cam_manager = CameraManager()
logger.debug("cam_manager loaded.")
poly_detector = SinglePolyDetector(SOURCE_IMG_PATH, SOURCE_JSON_PATH, target_label_name=TARGET_LABEL)
logger.debug("poly_detector loaded.")
ocr_engine = OcrEngine(OCR_MODEL_PATH)#, (2337, 100, 3))
logger.debug("ocr_engine loaded.")
# trainsmit_db = TransmitDB(DB_ADDR, DB_USER, DB_PASS)
# logger.debug("trainsmit_db loaded.")


def main(test_mode=False):
    logger.info(f"test mode : {test_mode}")
    print("To Exit, Press Ctrl+C")
    
    while True:
        time.sleep(LOOP_PERIOD)
        
        file_name = get_time_str() + ".jpg"
        
        img = cam_manager.get_image()
        poly_dict = poly_detector(img)
        if poly_dict is None:
            logger.info("no detect")
            path = os.path.join(RECODE_PATH, "no_detect", file_name)
            cv2.imwrite(path, img)
            cv2.imshow("test_show", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xff == ord('q'): break
            continue
        
        value_dict = {}
        for label in ["target_tmp", "actual_tmp"]:
            poly = poly_dict[label]
            crop_img, _ = get_crop_img_and_M(img, poly)
            crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
            pred_str = ocr_engine(crop_img)
            value_dict[label] = pred_str.strip()
        
        logger.info(f"value_dict : {value_dict}")
        
        # recode image
        img = draw_anno(img, poly_dict, value_dict)
        path = os.path.join(RECODE_PATH, "detect", file_name)
        cv2.imwrite(path, img)
        
        if test_mode:
            cv2.imshow("test_show", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xff == ord('q'): break
        else:
            if not value_dict["target_tmp"].isdigit(): value_dict["target_tmp"] = ''
            if not value_dict["actual_tmp"].isdigit(): value_dict["actual_tmp"] = ''
            trainsmit_db(value_dict)
            
    cv2.destroyAllWindows()
        

# def test():
#     path = "./temp/panel_rotate.png"
#     img = cv2.imread(path)
#     poly_dict = poly_detector(img)
    
#     for label in ["target_tmp", "actual_tmp"]:
#         poly = poly_dict[label]
#         crop_img, _ = get_crop_img_and_M(img, poly)
#         crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
#         pred_str = ocr_engine(crop_img)
#         print(pred_str)
#         cv2.imshow(label, crop_img)
#         cv2.waitKey(1)
        
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
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