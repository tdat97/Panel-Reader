# from utils.camera import Camera
from utils.poly import SinglePolyDetector, get_crop_img_and_M
from utils.ocr import OcrEngine
# from utils.transmit import TransmitDB

from utils.logger import logger
import time

# loop period (sec)
LOOP_PERIOD = 1

# Poly
SOURCE_IMG_PATH = "./source/panel2.png"
SOURCE_JSON_PATH = "./source/panel.json"
TARGET_LABEL = "panel"

# OCR
OCR_MODEL_PATH = "./source/OCR_aug3_300k.h5"
# OCR_MODEL_PATH = "./source/ocr_rgb.h5"

# Init
poly_detector = SinglePolyDetector(SOURCE_IMG_PATH, SOURCE_JSON_PATH, target_label_name=TARGET_LABEL)
logger.debug("poly_detector loaded.")
ocr_engine = OcrEngine(OCR_MODEL_PATH)#, (2337, 100, 3))
logger.debug("ocr_engine loaded.")

def main():    
    while True:
        time.sleep(LOOP_PERIOD)
        
        break

# test        
import cv2
def test():
    path = "./temp/panel_rotate.png"
    img = cv2.imread(path)
    poly_dict = poly_detector(img)
    
    for label in ["target_tmp", "actual_tmp"]:
        poly = poly_dict[label]
        crop_img, _ = get_crop_img_and_M(img, poly)
        crop_img = cv2.resize(crop_img, (0,0), fx=1.3, fy=1)
        pred_str = ocr_engine(crop_img)
        print(pred_str)
        cv2.imshow(label, crop_img)
        cv2.waitKey(1)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    test()