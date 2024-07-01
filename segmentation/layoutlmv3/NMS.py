import numpy as np

def non_max_suppression(boxes, scores, threshold):
    """
    Perform Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        boxes (List[List[float]]): List of bounding box coordinates in the format [x1, y1, x2, y2].
        scores (List[float]): List of confidence scores for each bounding box.
        threshold (float): Threshold for IoU (Intersection over Union) above which bounding boxes will be suppressed.
        
    Returns:
        List[int]: List of indices to keep after NMS.
    """
    assert len(boxes) == len(scores)

    if len(boxes) == 0:
        return []

    # Sort boxes by their scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        # Calculate IoU between the first box and all other boxes
        box_i = boxes[i]
        ious = [calculate_iou(box_i, boxes[j]) for j in sorted_indices[1:]]

        # Keep boxes with IoU less than threshold
        suppressed_indices = np.where(np.array(ious) <= threshold)[0]
        sorted_indices = sorted_indices[suppressed_indices + 1]

    return keep

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (List[float]): Coordinates of the first bounding box in the format [x1, y1, x2, y2].
        box2 (List[float]): Coordinates of the second bounding box in the format [x1, y1, x2, y2].
        
    Returns:
        float: IoU between the two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate precision and recall for object detection evaluation.
    
    Args:
        gt_boxes (List[List[float]]): List of ground truth bounding boxes in the format [x1, y1, x2, y2].
        pred_boxes (List[List[float]]): List of predicted bounding boxes in the format [x1, y1, x2, y2].
        iou_threshold (float): Intersection over Union (IoU) threshold for matching ground truth and predicted boxes.
        
    Returns:
        float: Precision
        float: Recall
    """
    tp = 0  # True positive
    fp = 0  # False positive
    fn = 0  # False negative

    # Convert lists to sets for faster lookup
    gt_set = set(tuple(box) for box in gt_boxes)
    pred_set = set(tuple(box) for box in pred_boxes)

    # Count true positives and false positives
    for pred_box in pred_set:
        match_found = False
        for gt_box in gt_set:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                match_found = True
                gt_set.remove(gt_box)
                break
        if not match_found:
            fp += 1

    # Count false negatives
    fn = len(gt_set)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall
