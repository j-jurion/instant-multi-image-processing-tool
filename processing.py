import cv2

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    _, thresholded = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    
    return thresholded
