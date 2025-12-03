import cv2

# Patch for OpenCV compatibility with ultralytics
if not hasattr(cv2, 'setNumThreads'):
    def setNumThreads(num_threads):
        pass
    cv2.setNumThreads = setNumThreads
