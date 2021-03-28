import os
import sys
import random
import itertools
import colorsys
import csv

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors) #Raju:
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3): # Raju:
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def PolygonArea(contours):
    n = len(contours)
    area = 0.0
    for i in range(n):
        j = (i+1)% n
        area += contours[i][0]*contours[j][1]
        area -= contours[j][0]*contours[i][1]
    area = abs(area)/2.0
    return area

def min_dis(a,b):
    # a = [np.array(i) for i in a]
    # b = [np.array(i) for i in b]
    minimum = min([np.linalg.norm(i-j) for i in a for j in b])
    return minimum


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask= True, show_bbox=False, #Raju: I changed "show_bbox = False", previously =True
                      colors=None, captions=False): #captions = None
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N) # Raju:

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('on') #off
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    data_list = list()
    data = dict()
    areas = [] # store the areas of every object
    distance1 = [] # to store the three categories of distance strings
    vertices1 = [] # store the vertices of masks
    mask1 = []

    for i in range(N):

        color = colors[i] # Raju:
        #color = colors[1] # Raju: I put this line

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:

            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none') #Raju: previously edgecolor = color
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i] #Raju:
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        # ax.text(x1, y1 + 2, caption, #Raju: 2 was 8
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        height_object = (x2- x1)
        width_object = (y2-y1)
        # area = height_object*width_object

        # print ("name : ", label , " : height is :", height_object , ", width is : ", width_object, "and area is :", area, " center " , x1 ,",", y1 )
        mask = masks[:, :, i]
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for items in contours:
            print('Calculating area:')
        vertices1.append(items)
        area = PolygonArea(items)
        areas.append(area)
        pier_1 = 1045
        pier_2 = 4632
        pier_3 = 8219

        pier_cap_1 = 467
        pier_cap_2 = 17492
        pier_cap_3 = 34500
        # globals()[label + '_1'] = pier_1
        # globals()[label + '_2'] = pier_2
        # globals()[label + '_3'] = pier_3

        if label == 'pier' and area > 10 and area < pier_2:
            distance = 'f'
        elif label == 'pier' and area > pier_2 and area < pier_3:
            distance ='m'
        elif label == 'pier' and area > pier_3:
            distance = 'c'
        elif label == 'pier cap' and area > pier_cap_1 and area < pier_cap_2:
            distance = 'f'
        elif label == 'pier cap' and area > pier_cap_2 and area < pier_cap_3:
            distance ='m'
        elif label == 'pier cap' and area > pier_cap_3:
            distance = 'c'
        else:
            distance = 'undefined'
        distance1.append(distance)
        mask1.append(mask)

        # data['label'] = label
        # data['area'] = area
        # data['distance'] = distance
        # # data['contours'] = items
        # print(data)
        #data_list.append(data)
    #print(areas)
    # print(vertices1)
    #print(distance1)
    #print(class_ids)
    #print(scores)
    #print(mask1)

    '''
    To-Do:
    Add one for loop iterating over len(class_id) to generate masks on top of
    the image. This for loop will be same as visualize file.
    add logics:
        1. Detection score
        2. based on the low detection score find the object from class_id list
         use enumerate function
        3. after getting the class id , find the distance from distance1 list,
         find the area and contours
        4. find the related object (e.g. for pier find pier cap), if related object
        is not there, then break
        5. find the distance for that object
        6. calculate minimum distance between this two objects
        7. if distance within a range then show mask otherwise suppress
    Lists are:
    areas, distance1, vertices1, class_ids, scores
    '''

    for x,y in enumerate(scores):
        nearby_pier_cap = 'yes'
        if y < 0.9:
            print ('x' , x, " = ", 'y ', y)
            class_low = class_ids[x] # finding the class of the low scoring object
            dis_type_low = distance1[x] #finding the distance type of the object
            area_low = areas[x]
            vertices_low = vertices1[x]
            #print('dis_type_low' , dis_type_low)
            if class_low == 7: #if the low scoring object is pier
                for a,b in enumerate (class_ids):
                    if b == 5: # locate the pier cap
                        dis_type_related = distance1[a]
                        vertices_related = vertices1[a]
                        if dis_type_related == dis_type_low:
                            minimum = min_dis(vertices_low,vertices_related)
                            print(minimum)
                            ## To-Do: Need to work on this: if min distance is more than
                            ## a certain value then it should check the next object
                            # if minimum > 100:
                            #      minimum = min_dis(vertices_low, vertices1[a])
                            # # break
                        else:
                            nearby_pier_cap = 'no'
                            print('didnot find a nearby pier cap')
                            continue
        # for k,m in enumerate(class_ids):

        print ('class_id' , class_id)
        if show_mask:
            if nearby_pier_cap == 'no':
                print('before continue')
            else:
                if class_id == 1:
                    masked_image = apply_mask(masked_image,mask1[x], [1.0, 0.0, 0.5454545454545459],alpha=0.5)
                elif class_id == 2:
                    print( 'this is slab')
                    masked_image = apply_mask(masked_image,mask1[x], [1.0, 0.5454545454545454, 0.0],alpha=0.5)
                elif class_id == 3:
                    masked_image = apply_mask(masked_image, mask1[x],[0.0, 0.7272727272727275, 1.0],alpha=0.5)
                elif class_id == 4:
                    masked_image = apply_mask(masked_image, mask1[x],[0.3636363636363633, 0.0, 1.0],alpha=0.5)
                elif class_id == 5:
                    masked_image = apply_mask(masked_image, mask1[x], [0.0, 0.18181818181818166, 1.0],alpha=0.5)
                elif class_id == 6:
                    masked_image = apply_mask(masked_image, mask1[x],[0.0, 1.0, 0.7272727272727271],alpha=0.5)
                elif class_id == 7:
                    masked_image = apply_mask(masked_image, mask1[x],[1.0, 0.0, 0.0],alpha=0.5)
                elif class_id == 8:
                    masked_image = apply_mask(masked_image, mask1[x],[0.36363636363636376, 1.0, 0.0],alpha=0.5)
                elif class_id == 9:
                    masked_image = apply_mask(masked_image, mask1[x],[0.9090909090909092, 0.0, 1.0],alpha=0.5)
                elif class_id == 10:
                    masked_image = apply_mask(masked_image, mask1[x],[0.9090909090909092, 1.0, 0.0],alpha=0.5)
            # elif class_id == 7:
            #     if nearby_pier_cap == 'no':
            #         print('before continue')
            #         # continue
            #     else:
            #         print('pier cap is nearby')
            #         masked_image = apply_mask(masked_image, mask1[x],[1.0, 0.0, 0.0],alpha=0.5)
            # elif class_id == 8:
            #     masked_image = apply_mask(masked_image, mask1[x],[0.36363636363636376, 1.0, 0.0],alpha=0.5)
            # elif class_id == 9:
            #     masked_image = apply_mask(masked_image, mask1[x],[0.9090909090909092, 0.0, 1.0],alpha=0.5)
            # elif class_id == 10:
            #     masked_image = apply_mask(masked_image, mask1[x],[0.9090909090909092, 1.0, 0.0],alpha=0.5)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


    # print(data_list)
        # print("area for :", label, 'is : ', area)

        # if i == 1:
        #     a = items
        #
        # elif i == 2:
        #     b = items
        #     #print ('a : ', a)
        #     #print ('b : ', b)
        #     minimum = min_dis(a,b)
        #     print('minimum :',minimum)
        # # minimum = min_dis(a,b)
        #
        # if i ==2:
        #     break
