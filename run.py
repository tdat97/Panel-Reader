from utils.camera import CameraManager
from utils.poly import SinglePolyDetector, get_crop_img_and_M
from utils.ocr import OcrEngine
# from utils.transmit import TransmitDB
from utils.logger import logger, switch_logger_level
from utils.tools import *

import time
import argparse

# loop period (sec)
LOOP_PERIOD = 1

# Poly
SOURCE_IMG_PATH = "./source/panel2.png"
SOURCE_JSON_PATH = "./source/panel.json"
TARGET_LABEL = "panel"

# OCR
OCR_MODEL_PATH = "./source/OCR_aug3_300k.h5"
# OCR_MODEL_PATH = "./source/ocr_rgb.h5"

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
    print("To Exit, Press Ctrl+C")
    while True:
        time.sleep(LOOP_PERIOD)
        
        image = cam_manager.get_image()
        poly_dict = poly_detector(img)
        if poly_dict is None: continue
        
        value_dict = {}
        for label in ["target_tmp", "actual_tmp"]:
            poly = poly_dict[label]
            crop_img, _ = get_crop_img_and_M(img, poly)
            crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
            pred_str = ocr_engine(crop_img)
            value_dict[label] = pred_str.strip()
        
        if not value_dict["target_tmp"].isdigit(): continue
        if not value_dict["actual_tmp"].isdigit(): continue
        value_dict["target_tmp"] = int(value_dict["target_tmp"])
        value_dict["actual_tmp"] = int(value_dict["actual_tmp"])
        
        if test_mode:
            image = draw_anno(image, poly_dict, value_dict)
            cv2.imshow("test_show", image)
            if cv2.waitKey(1) & 0xff == ord('q'): break
            
        # else:
        #     trainsmit_db(value_dict)
            
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