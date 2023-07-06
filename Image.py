import numpy as np
import cv2
from skimage.morphology import skeletonize

class Image:
    def __init__(self, imageName, asciiCode, image, skeleton=None, 
                 onlySkeleton=None):
        self.imageName = imageName
        self.asciiCode = asciiCode
        self.image = image
        if skeleton is None:
            self.skeleton = np.zeros(0)
        else:
            self.skeleton = skeleton
        if onlySkeleton is None:
            self.onlySkeleton = np.zeros(0)
        else:
            self.onlySkeleton = skeleton
    
    def getImageName(self):
        return self.imageName
    
    def getAsciiCode(self):
        return self.asciiCode
    
    def getSkeleton(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV)
        thresh = 1 * (thresh == 255)
        thresh = thresh.astype(np.uint8)
        skeleton = skeletonize(thresh, method='lee').astype(np.uint8)
        skeleton = skeleton * 255
        self.skeleton = skeleton
        return skeleton
    
    def getOnlySkeletonAndDims(self):
        if self.skeleton is None:
            print("There is no skeleton yet!!!")
            return 0, 0
        else:
            indices = np.argwhere(self.skeleton == 255)
            min_indices = np.min(indices, axis=0)
            max_indices = np.max(indices, axis=0)
            height = max_indices[0]-min_indices[0]+1
            width = max_indices[1]-min_indices[1]+1
            onlySkeleton = self.skeleton[min_indices[0]:max_indices[0]+1, 
                                         min_indices[1]:max_indices[1]+1]
            return onlySkeleton, width, height
    
    @staticmethod
    def rescaleSkeleton(skeleton, width, height):
        type = cv2.INTER_AREA
        skeletonResized = cv2.resize(skeleton, (width, height), interpolation=type)

        type = cv2.THRESH_BINARY+cv2.THRESH_OTSU
        ret2,thresh = cv2.threshold(skeletonResized, 0, 255, type=type)

        resized_skel_thresh = 1 * (thresh == 255)
        resized_skel_thresh = resized_skel_thresh.astype(np.uint8)
        rescaled_skeleton = skeletonize(resized_skel_thresh, method='lee')
        rescaled_skeleton = (rescaled_skeleton.astype(np.uint8)) * 255
        
        return rescaled_skeleton
