from simple_pyspin import Camera
import time
import cv2

ExposureAuto = 'Off'
ExposureTime = 10000
DeviceLinkThroughputLimit = 100 * 1024 * 1024
AcquisitionMode = "SingleFrame"

class CameraManager():
    def __init__(self, cam_number=0):
        self.cam = Camera(cam_number)
        self.cam.init()
        self.cam.ExposureAuto = ExposureAuto
        self.cam.ExposureTime = ExposureTime
        self.cam.DeviceLinkThroughputLimit = DeviceLinkThroughputLimit
        self.cam.AcquisitionMode = AcquisitionMode
        
        # warm up
        for _ in range(3):
            _ = self.get_image()
            time.sleep(0.1)
        
    def get_image(self):
        self.cam.start()
        image = self.cam.get_array()
        self.cam.stop()
        image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
        return image