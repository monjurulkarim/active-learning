def compute_precision_recall_spatial_correlation_temporal(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    areas = []
    distance1 = []
    vertices1 = []
    mask1 = []
    total_predicted = []
    safety_pxl = 30
    pxl = safety_pxl+12
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        if not np.any(pred_boxes[i]):
            continue
        y1, x1, y2, x2 = pred_boxes[i]
        score =pred_scores[i]
        class_id = pred_class_ids[i]
        mask = pred_masks[:, :, i]
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


        if class_id == 7 and area > 10 and area < pier_2:
            distance = 'f'
        elif class_id == 7 and area > pier_2 and area < pier_3:
            distance ='m'
        elif class_id == 7 and area > pier_3:
            distance = 'c'
        elif class_id == 5 and area > 10 and area < pier_cap_2:
            distance = 'f'
        elif class_id == 5 and area > pier_cap_2 and area < pier_cap_3:
            distance ='m'
        elif class_id == 5 and area > pier_cap_3:
            distance = 'c'
        else:
            distance = 'undefined'
        distance1.append(distance)
        mask1.append(mask)


    for x,y in enumerate(scores):
        nearby_pier_cap ='yes'
        class_id = pred_class_ids[x]
        if y <0.95:
            class_low = class_ids[x] # finding the class of the low scoring object
            dis_type_low = distance1[x] #finding the distance type of the object
            area_low = areas[x]
            vertices_low = vertices1[x]
            #print('dis_type_low' , dis_type_low)
            if class_low == 7: #if the low scoring object is pier
                for aa,bb in enumerate (class_ids):
                    if bb == 5: # locate the pier cap
                        dis_type_related = distance1[aa]
                        vertices_related = vertices1[aa]
                        if dis_type_related == dis_type_low:
                            minimum = min_dis(vertices_low,vertices_related)
                            print(minimum)
                            if minimum < 100:
                                print( 'there is nearby pier cap')
                                break
                            else:
                                nearby_pier_cap ='no'
                                break
                        else:
                            nearby_pier_cap = 'no'
                            print('didnot find a nearby pier cap')
                            continue
                    break
                    # else:
                    #     nearby_pier_cap = 'no'
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


        if nearby_pier_cap == 'yes':
            if f_ref == 'f1':
                if score >= 0.9:
                    total_predicted.append(score)
                elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)

                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)
                else:
                    continue

            if f_ref == 'f2':

                if score >= 0.9:
                    total_predicted.append(score)
                elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)
                else:
                    continue

            if f_ref == 'f3':
                if score >= 0.9:
                    total_predicted.append(score)
                elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

            if f_ref == 'f4':
                if score >= 0.9:
                    total_predicted.append(score)
                elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)
                else:
                    continue

            if f_ref == 'f5':
                if score >= 0.9:
                    total_predicted.append(score)
                elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)

                elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                    total_predicted.append(score)
                    cl.append(class_id)
                    Xco.append(x1)
                    Yco.append(y1)
                    # if not exists:
                    #     copyfile(src,dst+file)
                else:
                    continue
        else:
            continue

        sorted_ixs = np.argsort(overlaps[x])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[x, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[x, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                if nearby_pier_cap == 'yes':
                    if f_ref == 'f1':
                        if score >= 0.9:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                        elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break

                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)
                        else:
                            continue

                    if f_ref == 'f2':

                        if score >= 0.9:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                        elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)
                        else:
                            continue

                    if f_ref == 'f3':
                        if score >= 0.9:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                        elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in e and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                    if f_ref == 'f4':
                        if score >= 0.9:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                        elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in a and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)
                        else:
                            continue

                    if f_ref == 'f5':
                        if score >= 0.9:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                        elif score <0.9 and score >.50 and class_id in d and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in c and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)

                        elif score <0.9 and score >.50 and class_id in b and check(x1,Xcoordinate,pxl)==True and check_y(y1,Ycoordinate,pxl)== True:
                            match_count += 1
                            gt_match[j] = x
                            pred_match[i] = j
                            break
                            # if not exists:
                            #     copyfile(src,dst+file)
                        else:
                            continue
                else:
                    continue

    precision = match_count/len(pred_match)
    recall = match_count/len(gt_match)
    # f1 = 2*(precision*recall)/(precision+recall)
    f1 = 1
    csvfile = 'precision_recall.csv'
    file_exists = os.path.isfile(csvfile)
    myFile = open(csvfile, 'a', newline ='')
    with myFile:
        myFields = ['filename', 'true_positives', 'total_detected', 'ground_truth', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(myFile, fieldnames =myFields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'filename':file, 'true_positives': match_count, 'total_detected': len(total_predicted), 'ground_truth': len(gt_match), 'precision': precision, 'recall': recall, 'f1': f1 })

    return gt_match, pred_match, overlaps
