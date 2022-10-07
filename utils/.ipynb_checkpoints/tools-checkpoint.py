import cv2
import numpy as np

def draw_anno(img, poly_dict, value_dict):
    img = img.copy()
    
    # draw boxes
    polys = list(poly_dict.values())
    polys = np.stack(polys).astype(np.int32)
    cv2.polylines(img, polys, True, (255,0,255), 2)
    
    # write label
    for label, poly in zip(poly_dict.keys(), polys):
        left_top = poly[0]
        left_top[1] -= 10
        if label in value_dict: label += f" : {value_dict[label]}"
        cv2.putText(img, label, left_top, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    return img
    