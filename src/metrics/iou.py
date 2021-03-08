from src.utils.entities import BoundingBox


def compute_iou(bb1: BoundingBox, bb2: BoundingBox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    
    assert bb1.xtl < bb1.xbr
    assert bb1.ytl < bb1.ybr
    assert bb2.xtl < bb2.xbr
    assert bb2.ytl < bb2.ybr
    
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.xtl, bb2.xtl)
    y_top = max(bb1.ytl, bb2.ytl)
    x_right = min(bb1.xbr, bb2.xbr)
    y_bottom = min(bb1.ybr, bb2.ybr)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1.xbr - bb1.xtl) * (bb1.ybr - bb1.ytl)
    bb2_area = (bb2.xbr - bb2.xtl) * (bb2.ybr - bb2.ytl)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou