import json
import os
import random
import numpy as np  
import sys

from eval.utils import load_json, calc_iou

def calculate_accuracy(preds):
    correct_predictions = 0
    total_predictions = len(preds)
    
    for pred_entry in preds:
        if pred_entry['gt'] == pred_entry['predicted_id']:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

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
    easy_total = 0
    dep_total = 0
    correct_easy = 0
    correct_dep = 0

    correct_25 = 0
    unique_25 = 0
    correct_50 = 0
    
    print(len(pred_files))
    for pred_file in sorted(pred_files):

        correct_predictions = 0
        pred_file = os.path.join(pred_dir , pred_file)

        if not os.path.exists(pred_file): continue
        preds = load_json(pred_file)

        for pred_entry in preds:

            gt_bbox = pred_entry['gt_bbox']
            pred_bbox = pred_entry['pred_bbox']
            
            iou = calc_iou(gt_bbox, pred_bbox)

            if pred_entry['easy']:
                easy_total += 1
            if pred_entry['view_dep']:
                dep_total += 1

            if iou >= 0.25:
                if pred_entry['easy']:
                    correct_easy += 1
                if pred_entry['view_dep']:
                    correct_dep += 1

            if iou >= 0.25:
                correct_25 += 1

            if iou >= 0.5:
                correct_50 += 1

        total_predictions += len(preds)

    print()
    print()
    print('Easy     {:.2%}   {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
    print('Hard     {:.2%}   {} / {}'.format((correct_25 - correct_easy) / (total_predictions - easy_total),
                                correct_25 - correct_easy,
                                total_predictions - easy_total))
    print('Dep      {:.2%}   {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
    print('Indep    {:.2%}   {} / {}'.format((correct_25 - correct_dep) / (total_predictions - dep_total),
                                        correct_25 - correct_dep,
                                        total_predictions - dep_total))
    print()
    print('Acc@25           {:.2%}   {} / {}'.format(correct_25 / total_predictions, correct_25, total_predictions))
    print('Acc@50           {:.2%}   {} / {}'.format(correct_50 / total_predictions, correct_50, total_predictions))


if __name__ == '__main__':

    pred_dir = '/prediction_outputs/nr3d/agent-grounder/qwen3-vl-32b-instruct/validation_set_strats_nr3d_1000query/1/pred/'
    main(pred_dir)
