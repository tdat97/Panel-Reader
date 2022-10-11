import cv2
import numpy as np
import json


def json2label(path): # json 경로
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = [shape["label"] for shape in data["shapes"]]
    points = [np.array(shape["points"]).astype(np.int32)
             for shape in data["shapes"]] # (n, 2?, 2)
    return dict(zip(labels, points))

def get_poly_box_wh(poly_box): # (4, 2)
    lt, rt, rb, lb = poly_box
    w = int((np.linalg.norm(lt - rt) + np.linalg.norm(lb - rb)) // 2)
    h = int((np.linalg.norm(lt - lb) + np.linalg.norm(rt - rb)) // 2)
    return w, h

def crop_obj_in_bg(bg_img, polys):
    obj_imgs = []
    for poly in polys:
        poly = poly.astype(np.float32)
        w, h = get_poly_box_wh(poly)
        pos = np.float32([[0,0], [w,0], [w,h], [0,h]])
        M = cv2.getPerspectiveTransform(poly, pos)
        obj_img = cv2.warpPerspective(bg_img, M, (w, h))
        obj_imgs.append(obj_img)
    return obj_imgs

def get_crop_img_and_M(img, poly):
    poly = poly.astype(np.float32)
    w, h = get_poly_box_wh(poly)
    pos = np.float32([[0,0], [w,0], [w,h], [0,h]])
    M = cv2.getPerspectiveTransform(poly, pos)
    crop_img = cv2.warpPerspective(img, M, (w, h))
    return crop_img, M

class SinglePolyDetector():
    def __init__(self, img_path, json_path, pick_labels=[], n_features=5000):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img_gray is not None, "img_path is not correct."
        
        poly_dict = json2label(json_path)
        
        polys = [poly_dict[label] for label in pick_labels]
            
        # assert (target_label_name in self.labels) or (target_label_name==''), "Invalid label_name."
        
        # 0번 index를 target으로
        # target_idx = self.labels.index(target_label_name)
        # self.label[0], self.label[target_idx] = self.label[target_idx], self.label[0]
        # polys[0], polys[target_idx] = polys[target_idx], polys[0]
        
        # keypoints
        self.detector = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        crop_img_gray, M = get_crop_img_and_M(img_gray, polys[0])
        self.kp, self.desc = self.detector.detectAndCompute(crop_img_gray, None)
        
        # transform polygons
        polys = np.stack(polys).astype(np.float32)
        polys = cv2.perspectiveTransform(polys, M)
        self.src_polys = polys
        
    def __call__(self, img):
        if img.ndim == 2: img_gray = img
        else: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # match
        kp, desc = self.detector.detectAndCompute(img_gray, None)
        if len(kp) < 50: return None, None
        matches = self.matcher.match(self.desc, desc)
        
        # get keypoints of matches
        src_pts = np.float32([self.kp[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])
        
        # src_polys -> dst_polys
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 5.0)
        if mask.sum() / mask.size < 0.15: return None, None
        dst_polys = cv2.perspectiveTransform(self.src_polys, M)
        
        # get crop_imgs # 이래야 항상 크기가 일정함
        h, w = img.shape[:2]
        inv_M = cv2.getPerspectiveTransform(dst_polys[0], self.src_polys[0])
        img_trans = cv2.warpPerspective(img, inv_M, (w, h))
        crop_imgs = crop_obj_in_bg(img_trans, self.src_polys)
        
        return dst_polys, crop_imgs