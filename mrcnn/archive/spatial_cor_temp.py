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

def mask_color(class_id, masked_image,mask):
    if class_id == 1:
        masked_image = apply_mask(masked_image, mask, [1.0, 0.0, 0.5454545454545459],alpha=0.5)
    elif class_id == 2:
        masked_image = apply_mask(masked_image, mask, [1.0, 0.5454545454545454, 0.0],alpha=0.5)
    elif class_id == 3:
        masked_image = apply_mask(masked_image, mask,[0.0, 0.7272727272727275, 1.0],alpha=0.5)
    elif class_id == 4:
        masked_image = apply_mask(masked_image, mask,[0.3636363636363633, 0.0, 1.0],alpha=0.5)
    elif class_id == 5:
        masked_image = apply_mask(masked_image, mask, [0.0, 0.18181818181818166, 1.0],alpha=0.5)
    elif class_id == 6:
        masked_image = apply_mask(masked_image, mask,[0.0, 1.0, 0.7272727272727271],alpha=0.5)
    elif class_id == 7:
        masked_image = apply_mask(masked_image, mask,[1.0, 0.0, 0.0],alpha=0.5)
    elif class_id == 8:
        masked_image = apply_mask(masked_image, mask,[0.36363636363636376, 1.0, 0.0],alpha=0.5)
    elif class_id == 9:
        masked_image = apply_mask(masked_image, mask,[0.9090909090909092, 0.0, 1.0],alpha=0.5)
    elif class_id == 10:
        masked_image = apply_mask(masked_image, mask,[0.9090909090909092, 1.0, 0.0],alpha=0.5)
    return

def check(x1,Xcoordinate,pxl):
    if Xcoordinate-pxl <= x1 <= Xcoordinate + pxl:
        return True
    return False

def check_y(y1, Ycoordinate, pxl):
    if Ycoordinate-pxl <= y1 <= Ycoordinate + pxl:
        return True
    return False

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
    minimum = min([np.linalg.norm(i-j) for i in a for j in b])
    return minimum


def display_instances(f_ref,image,file,filename,filesize,json1, boxes, masks, class_ids, class_names,a,b,c,d,e,cl,Xco,Yco,Xcor1,Ycor1,Xcor2,Ycor2,Xcor3,Ycor3,Xcor4,Ycor4,Xcor5,Ycor5,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=False, #Raju: I changed "show_bbox = False", previously =True
                      colors=None, captions=None): #captions = None
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
    safety_pxl = 40
    # data_list = list()
    # data = dict()
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
        # if show_bbox:
        #
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none') #Raju: previously edgecolor = color
        #     ax.add_patch(p)


        mask = masks[:, :, i]
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for items in contours: vertices1.append(items)

        area = PolygonArea(items)
        areas.append(area)
        pier_1 = 1045
        pier_2 = 4632
        pier_3 = 8219

        pier_cap_1 = 467
        pier_cap_2 = 17492
        pier_cap_3 = 34500


        if class_ids[i] == 7 and area > 10 and area < pier_2:
            distance = 'f'
        elif class_ids[i] == 7 and area > pier_2 and area < pier_3:
            distance ='m'
        elif class_ids[i] == 7 and area > pier_3:
            distance = 'c'
        elif class_ids[i] == 5 and area > 10 and area < pier_cap_2:
            distance = 'f'
        elif class_ids[i] == 5 and area > pier_cap_2 and area < pier_cap_3:
            distance ='m'
        elif class_ids[i] == 5 and area > pier_cap_3:
            distance = 'c'
        else:
            distance = 'undefined'
        distance1.append(distance)
        mask1.append(mask)


    print('file : ', file)
    print('class_ids :', class_ids)
    print('scores :', scores)
    print('distance_type : ', distance1)




    for x,y in enumerate(scores):
        nearby_pier_cap = 'yes'
        class_id = class_ids[x]
        if y < 0.95:
            class_low = class_ids[x] # finding the class of the low scoring object
            print('class_low : ', class_low)
            dis_type_low = distance1[x] #finding the distance type of the object
            area_low = areas[x]
            vertices_low = vertices1[x]
            #print('dis_type_low' , dis_type_low)
            if class_low == 7: #if the low scoring object is pier
                for aa,bb in enumerate (class_ids):
                    print('aa :', aa, '-->', ' bb :', bb)
                    if bb == 5: # locate the pier cap
                        dis_type_related = distance1[aa]
                        print('class_id : ', class_ids[x])
                        print('distance_type_related: ', dis_type_related)
                        print('distance_type_low: ', dis_type_low)
                        print('=====')
                        vertices_related = vertices1[aa]
                        if dis_type_related == dis_type_low:
                            minimum = min_dis(vertices_low,vertices_related)
                            print(minimum)
                            if minimum < 100:
                                print( 'there is nearby pier cap')
                                break
                            else:
                                print('else no_pier cap')
                                nearby_pier_cap ='no'
                        else:
                            nearby_pier_cap = 'no'
                            print('didnot find a nearby pier cap')
                            continue
                    break
            elif class_low == 5:
                for aa,bb in enumerate (class_ids):
                    print('aa :', aa, '-->', ' bb :', bb)
                    if bb == 7: # locate the pier cap
                        dis_type_related = distance1[aa]
                        print('class_id : ', class_ids[x])
                        print('distance_type_related: ', dis_type_related)
                        print('distance_type_low: ', dis_type_low)
                        print('=====')
                        vertices_related = vertices1[aa]
                        if dis_type_related == dis_type_low:
                            minimum = min_dis(vertices_low,vertices_related)
                            print(minimum)
                            if minimum < 100:
                                print( 'there is nearby pier')
                                break
                            else:
                                print('else no_pier cap')
                                nearby_pier_cap ='no'
                        else:
                            nearby_pier_cap = 'no'
                            print('didnot find a nearby pier cap')
                            continue
                    break
                    # else:
                    #     print('for scores: ', y)
                    #     print('it is going inside else')
                    #     nearby_pier_cap = 'no'

        '''
        The below part is for temporal coherence

        '''
        Xcoordinate = 0
        Ycoordinate = 0
        Xcoordinate1 = 0
        Ycoordinate1 = 0
        pxl = 0

        if f_ref == 'f1':
            if class_ids[x] in e:
                for m,n in enumerate(e):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor5[m]
                        Ycoordinate = Ycor5[m]
                        pxl = safety_pxl +3

            elif class_ids[x] in d:
                for m,n in enumerate(d):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor4[m]
                        Ycoordinate = Ycor4[m]
                        pxl = safety_pxl+6

            elif class_ids[x] in c:
                for m,n in enumerate(c):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor3[m]
                        Ycoordinate = Ycor3[m]
                        pxl = safety_pxl +9


        if f_ref == 'f2':
            # print('f2: class_id =',class_ids[x])
            # print ('a :', a)
            # print ('e : ', e)
            if class_ids[x] in a:
                for m,n in enumerate(a):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor1[m]
                        Ycoordinate = Ycor1[m]
                        pxl = safety_pxl +3

            elif class_ids[x] in e:
                for m,n in enumerate(e):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor5[m]
                        Ycoordinate = Ycor5[m]
                        pxl = safety_pxl + 6

            elif class_ids[x] in d:
                for m,n in enumerate(d):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor4[m]
                        Ycoordinate = Ycor4[m]
                        pxl = safety_pxl + 9


        if f_ref == 'f3':
            if class_ids[x] in b:
                for m,n in enumerate(b):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor2[m]
                        Ycoordinate = Ycor2[m]
                        pxl = safety_pxl +3

            elif class_ids[x] in a:
                for m,n in enumerate(a):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor1[m]
                        Ycoordinate = Ycor1[m]
                        pxl = safety_pxl + 6

            elif class_ids[x] in e:
                for m,n in enumerate(e):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor5[m]
                        Ycoordinate = Ycor5[m]
                        pxl = safety_pxl + 9

        if f_ref == 'f4':
            if class_ids[x] in c:
                for m,n in enumerate(c):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor3[m]
                        Ycoordinate = Ycor3[m]
                        pxl = safety_pxl +3

            elif class_ids[x] in b:
                for m,n in enumerate(b):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor2[m]
                        Ycoordinate = Ycor2[m]
                        pxl = safety_pxl +6

            elif class_ids[x] in a:
                for m,n in enumerate(a):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor1[m]
                        Ycoordinate = Ycor1[m]
                        pxl = safety_pxl +9

        if f_ref == 'f5':
            if class_ids[x] in d:
                for m,n in enumerate(d):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor4[m]
                        Ycoordinate = Ycor4[m]
                        pxl = safety_pxl +3

            elif class_ids[x] in c:
                for m,n in enumerate(c):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor3[m]
                        Ycoordinate = Ycor3[m]
                        pxl = safety_pxl +6

            elif class_ids[x] in b:
                for m,n in enumerate(b):
                    if n == class_ids[x]:
                        Xcoordinate = Xcor2[m]
                        Ycoordinate = Ycor2[m]
                        pxl = safety_pxl +9


        y1, x1, y2, x2 = boxes[x]
        if not captions:
            score = scores[x]
            label = class_names[class_id]
            # x = random.randint(x1, (x1 +x2)//2)
            caption = "{}{:.3f}".format(label,score) if score else label
        else:
            caption = caption[x]

        mask = mask1[x]
        ax.text(x1,y1+2, caption, color = 'w', size =11, backgroundcolor ='black')

        if show_mask:
            if nearby_pier_cap == 'no':
                print('before continue')
            else:
                # print('filename : ', file)
                # print('f_ref : ', f_ref)
                # print('class_id : ', class_id, ' class_id[x] : ' , class_ids[x] )
                # print('label : ', label)
                # print('score : ', score)
                # print('a : ', a)
                # print('b : ', b)
                # print('c : ', c)
                # print('d : ', d)
                # print('e : ', e)
                # print('pxl : ', pxl)
                # print('x1 : ', x1)
                # print('Xcordinate : ', Xcoordinate)
                # print('y1 : ', y1)
                # print('Ycordinate : ', Ycoordinate)
                if f_ref == 'f1':
                    if score >= 0.95:
                        mask_color(class_id, masked_image, mask)
                    elif score <0.95 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)

                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id,masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id,masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)
                    else:
                        continue

                if f_ref == 'f2':

                    if score >= 0.95:
                        mask_color(class_id, masked_image, mask)
                    elif score <0.95 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)
                    else:
                        continue

                if f_ref == 'f3':
                    if score >= 0.95:
                        mask_color(class_id,masked_image, mask)
                    elif score <0.95 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id,masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id,masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                if f_ref == 'f4':
                    if score >= 0.95:
                        mask_color(class_id,masked_image, mask)
                    elif score <0.95 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id,masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)
                    else:
                        continue

                if f_ref == 'f5':
                    if score >= 0.95:
                        mask_color(class_id, masked_image, mask)
                    elif score <0.95 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)

                    elif score <0.95 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                        mask_color(class_id, masked_image, mask)
                        cl.append(class_id)
                        Xco.append(x1)
                        Yco.append(y1)
                        # if not exists:
                        #     copyfile(src,dst+file)
                    else:
                        continue
                #
                #
                # ax.text(x1,y1+2, caption, color = 'w', size =11, backgroundcolor ='none')
                # mask_color(class_id, masked_image,mask1[x])
    print('******')

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
