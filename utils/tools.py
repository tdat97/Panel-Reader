import cv2
import numpy as np
import datetime
from glob import glob
import os


def draw_anno(img, labels, polys, values):
    assert len(labels) == len(polys) == len(values)
    
    img = img.copy()
    polys = polys.astype(np.int32)
    cv2.polylines(img, polys, True, (255,0,255), 2)
    
    for label, poly, value in zip(labels, polys, values):
        left_top = poly[0]
        left_top[1] -= 10
        text = f"{label} : {value}" if value else label
        cv2.putText(img, text, left_top, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    return img
    
    
def get_time_str():
    now = datetime.datetime.now()
    s = f"{now.year:04d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}"
    s += f"_{now.microsecond:06d}"
    return s

def manage_file_num(dir_path, max_size=1000, num_remove=100):
    path = os.path.join(dir_path, "*.jpg")
    img_paths = sorted(glob(path))
    if len(img_paths) < max_size: return

    for path in img_paths[:num_remove]:
        os.remove(path)