
import json
import numpy as np  

def load_json(pred_file):
    with open(pred_file, 'r') as f:
        return json.load(f)

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    if box_b is None:
        return 0
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    try:
        max_a = box_a[0:3] + box_a[3:6] / 2
        max_b = box_b[0:3] + box_b[3:6] / 2
    except:
        import pdb
        pdb.set_trace()
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union