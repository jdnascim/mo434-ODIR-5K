import cv2 
import os
import numpy as np

# create a CLAHE with L channel(Contrast Limited Adaptive Histogram Equalization).
def pre_proc_CEH2(img): 
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cl1 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cl1

# create a CLAHE (Contrast Limited Adaptive Histogram Equalization).	
def pre_proc_CEH(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize	)
    cl1 = clahe.apply(img_bw)
    return cl1

# create Equalization Histogram
def pre_proc_EH(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_bw)
    eh1 = np.hstack((img_bw,equ))
    return eh1

# reduce the black background
def cut_img(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(grayscale, 0, 255,cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    img_cut = img[y:y+h, x:x+w]    
    return img_cut

# reduce the black background
def cut_and_resize_to_original_img(img):
    shp = img.shape[0:2]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , thresholded = cv2.threshold(grayscale, 0, 255,cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    img_cut = img[y:y+h, x:x+w]    
    bgr_final = cv2.cvtColor(img_cut, cv2.COLOR_LAB2BGR)
    img_cut_resized = cv2.resize(bgr_final,shp,interpolation=cv2.INTER_AREA)
    return img_cut_resized

def CEH_cut_pipeline(img):
    img_uint = img.astype(np.uint8)
    img1 = cut_and_resize_to_original_img(img_uint)
    img2 = pre_proc_CEH2(img1)
    return img2

# Read image per image
def load_images_from_folder(path_folder): 
    PROC_FOLDER = path_folder + "_procEH/"
    if os.path.isdir(os.path.dirname(PROC_FOLDER)) is False:
        os.makedirs(os.path.dirname(PROC_FOLDER))

    for filename in os.listdir(path_folder):
        img = cv2.imread(os.path.join(path_folder,filename))
        if img is not None:
            img_proc = cut_img(img)
            img_proc = pre_proc_EH(img) #change with pre_proc_EH
            path = os.path.join(PROC_FOLDER, filename)
            cv2.imwrite(path, img_proc)

# CHANGE THE DIRECTORY OF IMAGES
#load_images_from_folder("test2")