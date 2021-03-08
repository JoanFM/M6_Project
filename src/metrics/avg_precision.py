from src.metrics.iou import compute_iou
from typing import List
from src.utils.entities import BoundingBox

def compute_average_precision(mIOU: float, groundtruths_bbox: List[BoundingBox], resulting_bbox = List[BoundingBox]):
    results = []
    for gt_bbox, result_bbox in zip(groundtruths_bbox, resulting_bbox):
        if compute_iou(gt_bbox, result_bbox) >= mIOU:
            results.append(True)
        else:
            results.append(False)
            
    def precision(i):
        count = len(list(filter(lambda x: x is True, results[:i])))
        return count / i
        
    def recall(i):
        count = len(list(filter(lambda x: x is True, results[:i])))
        return count / len(gt_bbox)
    
    ap = 0
    for i in range(1, len(results)):
        ap += (recall(i) - recall(i - 1))* precision(i)
    
    return ap