import cv2 
import os

def pre_proc(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 0, 255,cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    print(bbox)
    img_cut = img[y:y+h, x:x+w] 
    img_cut_bw = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    
    # create a CLAHE (Contrast Limited Adaptive Histogram Equalization).  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_cut_bw)
    return cl1

def load_images_from_folder(path_folder): 
    PROC_FOLDER = path_folder + "_proc/"
    if os.path.isdir(os.path.dirname(PROC_FOLDER)) is False:
        os.makedirs(os.path.dirname(PROC_FOLDER))

    for filename in os.listdir(path_folder):
        img = cv2.imread(os.path.join(path_folder,filename))
        if img is not None:
            img_proc = pre_proc(img)
            path = os.path.join(PROC_FOLDER, filename)
            print(path)
            cv.imwrite(path, img_proc)

# CHANGE THE DIRECTORY OF IMAGES
load_images_from_folder("train")