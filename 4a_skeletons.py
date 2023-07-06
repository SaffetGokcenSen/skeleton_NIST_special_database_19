from Image import Image
import os
import cv2
import numpy as np

# the location where train images for 4a (J) character reside
path = (
    "/home/saffetgokcen/Documents/Causal Inference/latex/causalImageMnist/"
    "train_images/train_4a"
)
file_iterator = os.scandir(path=path)

resultsPath = (
    "/home/saffetgokcen/Documents/Causal Inference/latex/causalImageMnist/"
    "Python/object_oriented/rescaled_skeleton_4a/"
)

i = 0
# image names with width and height greater than 28 pixels
dataSetImages = []
heightList = []
widthList = []
for file_name in file_iterator:
    # name of the image file
    image_name = file_name.name
    # path to the location of the image
    image_path = path + '/' + image_name
    # read the image
    image = cv2.imread(image_path)

    # create the image object
    imageObj = Image(image_name, "4a", image)

    # the skeleton of the image is obtained. in this image, the skeleton and 
    # redundant pixels exist.
    imgSkeleton = imageObj.getSkeleton()

    # onlySkeleton image contains only the skeleton, no redundant pixels.
    # width and height belong to the onlySkeleton image.
    onlySkeleton, width, height = imageObj.getOnlySkeletonAndDims()

    # if the width and height of the onlySkeleton is greater than or equal to 28 
    # pixels, rescale onlySkeleton so that it will have a width and height of 
    # 28 pixels.
    if (width>=28) & (height>=28):

        rescaledOnlySkeleton = imageObj.rescaleSkeleton(onlySkeleton, 28, 28)

        indices = np.argwhere(rescaledOnlySkeleton == 255)
        min_indices = np.min(indices, axis=0)
        max_indices = np.max(indices, axis=0)
        height = max_indices[0]-min_indices[0]+1
        width = max_indices[1]-min_indices[1]+1

        if ((np.absolute(28-width) <= 1) & (np.absolute(28-height) <= 1)):
            heightList.append(height)
            widthList.append(width)
            dataSetImages.append(imageObj.getImageName())
            resultName = 'skeleton_rescale_' + image_name
            _ = cv2.imwrite(resultsPath + resultName, rescaledOnlySkeleton)

widthArray = np.array(widthList)
heightArray = np.array(heightList)

import matplotlib.pyplot as plt
def dimensionHistogram(histType, theArray, thePath, theTitle):
    fig, ax = plt.subplots()
    histAuto, binEdgesAuto, _ = plt.hist(theArray, bins=histType)
    _ = ax.set_title(theTitle)
    _ = fig.savefig(thePath)
    _ = plt.close()

discPath = (
    "/home/saffetgokcen/Documents/Causal Inference/latex/causalImageMnist/"
    "Python/object_oriented/histogram_4a_width.png"
    )
theTitle = "4a, rescaled skeleton width"
dimensionHistogram('auto', widthArray, discPath, theTitle)

discPath = (
    "/home/saffetgokcen/Documents/Causal Inference/latex/causalImageMnist/"
    "Python/object_oriented/histogram_4a_height.png"
    )
theTitle = "4a, rescaled skeleton height"
dimensionHistogram('auto', heightArray, discPath, theTitle)

import json
discPath = (
    "/home/saffetgokcen/Documents/Causal Inference/latex/causalImageMnist/"
    "Python/object_oriented/dataSet4aImageNames.json"
    )
with open(discPath, 'w') as f:
    json.dump(dataSetImages, f, indent=2)
