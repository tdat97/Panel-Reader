from simple_pyspin import Camera
import time

ExposureTime = 10000
DeviceLinkThroughputLimit = 100 * 1024 * 1024
AcquisitionMode = "SingleFrame"

class CameraManager():
    def __init__(self, cam_number=0, exposure_time=ExposureTime, \
                 device_link_throughput_limit=DeviceLinkThroughputLimit, \
                 acquisition_mode=AcquisitionMode):
        self.cam = Camera(cam_number)
        self.cam.init()
        self.cam.ExposureTime = exposure_time
        self.cam.DeviceLinkThroughputLimit = device_link_throughput_limit
        self.cam.AcquisitionMode = acquisition_mode
        
        # warm up
        for _ in range(3):
            _ = self.get_image()
            time.sleep(0.1)
        
    def get_image(self):
        self.cam.start()
        image = cam.get_array()
        self.cam.stop()
        return image