def compute_precision_recall_spatial_correlation(gt_boxes, gt_class_ids, gt_masks,
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


        if label == 'pier' and area > 10 and area < pier_2:
            distance = 'f'
        elif label == 'pier' and area > pier_2 and area < pier_3:
            distance ='m'
        elif label == 'pier' and area > pier_3:
            distance = 'c'
        elif label == 'pier cap' and area > 10 and area < pier_cap_2:
            distance = 'f'
        elif label == 'pier cap' and area > pier_cap_2 and area < pier_cap_3:
            distance ='m'
        elif label == 'pier cap' and area > pier_cap_3:
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
                for a,b in enumerate (class_ids):
                    if b == 5: # locate the pier cap
                        dis_type_related = distance1[a]
                        vertices_related = vertices1[a]
                        if dis_type_related == dis_type_low:
                            minimum = min_dis(vertices_low,vertices_related)
                            print(minimum)
                            if minimum < 100:
                                print( 'there is nearby pier cap')
                            else:
                                nearby_pier_cap ='no'
                        else:
                            nearby_pier_cap = 'no'
                            print('didnot find a nearby pier cap')
                            continue
                    else:
                        nearby_pier_cap = 'no'
        if nearby_pier_cap == 'yes':
            total_predicted.append(score)
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
                    match_count += 1
                    gt_match[j] = x
                    pred_match[i] = j
                    break
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
