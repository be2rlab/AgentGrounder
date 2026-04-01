import json
import os
import random
import numpy as np  
import sys
from eval.utils import load_json, calc_iou

def main(pred_dir):
    """
    Evaluate the accuracy of prediction files in a directory. The directory should contain JSON files with prediction data.
    """

    pred_files = [f for f in os.listdir(pred_dir ) if f.endswith('.json')]
    print(f'Found {len(pred_files)} JSON files in {pred_dir }')
    assert len(pred_files) > 0, 'No JSON files found in the specified directory'

    total_correct_predictions = 0
    total_predictions = 0
    unique_total = 0

    correct_25 = 0
    unique_25 = 0
    correct_50 = 0
    unique_50 = 0

    for pred_file in pred_files:

        correct_predictions = 0
        pred_file = os.path.join(pred_dir , pred_file)
        preds = load_json(pred_file)

        for pred_entry in preds:

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']
            
            try:
                iou = calc_iou(gt_bbox, pred_bbox)
            except:
                import pdb
                pdb.set_trace()

            if pred_entry['unique']:
                unique_total += 1

            if iou >= 0.25:
                correct_25 += 1
                if pred_entry['unique']:
                    unique_25 += 1

            if iou >= 0.5:
                correct_50 += 1
                if pred_entry['unique']:
                    unique_50 += 1

        total_predictions += len(preds)

    print()

    print('Unique@25        {:.2%}   {} / {}'.format(unique_25 / unique_total, unique_25, unique_total))
    print('Multiple@25      {:.2%}    {} / {}'.format((correct_25 - unique_25) / (total_predictions - unique_total), correct_25 - unique_25, total_predictions - unique_total))
    print('Unique@50        {:.2%}   {} / {}'.format(unique_50 / unique_total, unique_50, unique_total))
    print('Multiple@50      {:.2%}    {} / {}'.format((correct_50 - unique_50) / (total_predictions - unique_total), correct_50 - unique_50, total_predictions - unique_total))
    print()
    print('Acc@25           {:.2%}   {} / {}'.format(correct_25 / total_predictions, correct_25, total_predictions))
    print('Acc@50           {:.2%}   {} / {}'.format(correct_50 / total_predictions, correct_50, total_predictions))


if __name__ == '__main__':
    pred_dir = '/prediction_outputs/scanrefer/agent-grounder/qwen3-vl-32b-instruct/validation_set_strats_scanrefer_1000query/1/pred/'
    main(pred_dir)
