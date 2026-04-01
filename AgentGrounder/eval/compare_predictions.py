"""
Compare two prediction versions for 3D Visual Grounding evaluation.
Analyzes differences between two sets of predictions based on IoU thresholds.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

from utils import load_json, calc_iou


def compare_predictions(dir_a, dir_b, iou_threshold=0.25, name_a="Version A", name_b="Version B"):
    """
    Compare predictions from two directories.
    
    Categories:
    - both_correct: Both versions have IoU >= threshold
    - both_wrong: Both versions have IoU < threshold
    - only_a_correct: Only version A is correct
    - only_b_correct: Only version B is correct
    - same_prediction: Both predict the same object ID
    - diff_prediction: Predict different object IDs
    """
    
    files_a = set(f for f in os.listdir(dir_a) if f.endswith('.json'))
    files_b = set(f for f in os.listdir(dir_b) if f.endswith('.json'))
    
    common_files = sorted(files_a & files_b)
    only_in_a = sorted(files_a - files_b)
    only_in_b = sorted(files_b - files_a)
    
    print(f"{'='*80}")
    print(f"Comparing: {name_a} vs {name_b}")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"{'='*80}")
    print(f"Common files: {len(common_files)}")
    if only_in_a:
        print(f"Only in {name_a}: {len(only_in_a)} files")
    if only_in_b:
        print(f"Only in {name_b}: {len(only_in_b)} files")
    print()
    
    # Results containers
    both_correct = []
    both_wrong = []
    only_a_correct = []
    only_b_correct = []
    same_pred = []
    diff_pred = []
    
    # Per-scene stats
    scene_stats = {}
    
    total = 0
    
    for pred_file in common_files:
        preds_a = load_json(os.path.join(dir_a, pred_file))
        preds_b = load_json(os.path.join(dir_b, pred_file))
        
        scene_name = pred_file.replace('.json', '')
        scene_stats[scene_name] = {
            'both_correct': 0, 'both_wrong': 0,
            'only_a': 0, 'only_b': 0, 'total': 0
        }
        
        # Match by query + gt_id
        index_b = {}
        for i, entry in enumerate(preds_b):
            query = entry['query'].replace(' .', '.').replace(' ,', ',').strip()
            key = (query, entry['gt_id'])
            index_b[key] = entry
        
        for entry_a in preds_a:
            query = entry_a['query'].replace(' .', '.').replace(' ,', ',').strip()
            key = (query, entry_a['gt_id'])
            entry_b = index_b.get(key)
            
            if entry_b is None:
                continue
            
            total += 1
            scene_stats[scene_name]['total'] += 1
            
            iou_a = calc_iou(entry_a['gt_bbox'], entry_a['pred_bbox'])
            iou_b = calc_iou(entry_b['gt_bbox'], entry_b['pred_bbox'])
            
            correct_a = iou_a >= iou_threshold
            correct_b = iou_b >= iou_threshold
            
            pred_id_a = entry_a.get('predicted_id')
            pred_id_b = entry_b.get('predicted_id')
            same = (pred_id_a == pred_id_b)
            
            record = {
                'scene': scene_name,
                'query': entry_a['query'][:100],
                'gt_id': entry_a['gt_id'],
                'pred_id_a': pred_id_a,
                'pred_id_b': pred_id_b,
                'iou_a': iou_a,
                'iou_b': iou_b,
                'unique': entry_a.get('unique', None),
            }
            
            if same:
                same_pred.append(record)
            else:
                diff_pred.append(record)
            
            if correct_a and correct_b:
                both_correct.append(record)
                scene_stats[scene_name]['both_correct'] += 1
            elif not correct_a and not correct_b:
                both_wrong.append(record)
                scene_stats[scene_name]['both_wrong'] += 1
            elif correct_a and not correct_b:
                only_a_correct.append(record)
                scene_stats[scene_name]['only_a'] += 1
            else:
                only_b_correct.append(record)
                scene_stats[scene_name]['only_b'] += 1
    
    # ===== Summary =====
    print(f"{'='*80}")
    print(f"OVERALL SUMMARY (IoU >= {iou_threshold})")
    print(f"{'='*80}")
    print(f"Total matched queries: {total}")
    print()
    
    print(f"  Both correct:        {len(both_correct):>5}  ({len(both_correct)/total:.2%})")
    print(f"  Both wrong:          {len(both_wrong):>5}  ({len(both_wrong)/total:.2%})")
    print(f"  Only {name_a} correct: {len(only_a_correct):>5}  ({len(only_a_correct)/total:.2%})")
    print(f"  Only {name_b} correct: {len(only_b_correct):>5}  ({len(only_b_correct)/total:.2%})")
    print()
    print(f"  Same prediction ID:  {len(same_pred):>5}  ({len(same_pred)/total:.2%})")
    print(f"  Diff prediction ID:  {len(diff_pred):>5}  ({len(diff_pred)/total:.2%})")
    print()
    
    acc_a = (len(both_correct) + len(only_a_correct)) / total
    acc_b = (len(both_correct) + len(only_b_correct)) / total
    print(f"  {name_a} accuracy:    {acc_a:.2%}  ({len(both_correct) + len(only_a_correct)} / {total})")
    print(f"  {name_b} accuracy:    {acc_b:.2%}  ({len(both_correct) + len(only_b_correct)} / {total})")
    print()
    
    # ===== Unique vs Multiple breakdown =====
    unique_records = [r for r in both_correct + both_wrong + only_a_correct + only_b_correct if r.get('unique')]
    multi_records = [r for r in both_correct + both_wrong + only_a_correct + only_b_correct if not r.get('unique')]
    
    if unique_records:
        u_total = len(unique_records)
        u_both_c = sum(1 for r in unique_records if r in both_correct)
        u_both_w = sum(1 for r in unique_records if r in both_wrong)
        u_only_a = sum(1 for r in unique_records if r in only_a_correct)
        u_only_b = sum(1 for r in unique_records if r in only_b_correct)
        
        print(f"  --- Unique queries ({u_total}) ---")
        print(f"    Both correct:        {u_both_c:>5}  ({u_both_c/u_total:.2%})")
        print(f"    Both wrong:          {u_both_w:>5}  ({u_both_w/u_total:.2%})")
        print(f"    Only {name_a} correct: {u_only_a:>5}  ({u_only_a/u_total:.2%})")
        print(f"    Only {name_b} correct: {u_only_b:>5}  ({u_only_b/u_total:.2%})")
        print()
    
    if multi_records:
        m_total = len(multi_records)
        m_both_c = sum(1 for r in multi_records if r in both_correct)
        m_both_w = sum(1 for r in multi_records if r in both_wrong)
        m_only_a = sum(1 for r in multi_records if r in only_a_correct)
        m_only_b = sum(1 for r in multi_records if r in only_b_correct)
        
        print(f"  --- Multiple queries ({m_total}) ---")
        print(f"    Both correct:        {m_both_c:>5}  ({m_both_c/m_total:.2%})")
        print(f"    Both wrong:          {m_both_w:>5}  ({m_both_w/m_total:.2%})")
        print(f"    Only {name_a} correct: {m_only_a:>5}  ({m_only_a/m_total:.2%})")
        print(f"    Only {name_b} correct: {m_only_b:>5}  ({m_only_b/m_total:.2%})")
        print()
    
    # ===== Cases where only one version is correct =====
    if only_a_correct:
        print(f"\n{'='*80}")
        print(f"REGRESSIONS: Only {name_a} correct, {name_b} wrong ({len(only_a_correct)} cases)")
        print(f"{'='*80}")
        # for r in sorted(only_a_correct, key=lambda x: x['iou_a'] - x['iou_b'], reverse=True)[:20]:
        for r in sorted(only_a_correct, key=lambda x: x['scene']): # [:20]:
            print(f"  [{r['scene']}] gt={r['gt_id']} | "
                  f"{name_a}: pred={r['pred_id_a']} IoU={r['iou_a']:.4f} | "
                  f"{name_b}: pred={r['pred_id_b']} IoU={r['iou_b']:.4f} | "
                  f"unique={r['unique']}")
            print(f"    query: {r['query']}...")
        # if len(only_a_correct) > 20:
        #     print(f"  ... and {len(only_a_correct) - 20} more")
    
    if only_b_correct:
        print(f"\n{'='*80}")
        print(f"IMPROVEMENTS: Only {name_b} correct, {name_a} wrong ({len(only_b_correct)} cases)")
        print(f"{'='*80}")
        # for r in sorted(only_b_correct, key=lambda x: x['iou_b'] - x['iou_a'], reverse=True)[:20]:
        for r in sorted(only_b_correct, key=lambda x: x['scene']):
            print(f"  [{r['scene']}] gt={r['gt_id']} | "
                  f"{name_a}: pred={r['pred_id_a']} IoU={r['iou_a']:.4f} | "
                  f"{name_b}: pred={r['pred_id_b']} IoU={r['iou_b']:.4f} | "
                  f"unique={r['unique']}")
            print(f"    query: {r['query']}...")
        # if len(only_b_correct) > 20:
        #     print(f"  ... and {len(only_b_correct) - 20} more")
    
    # ===== Diff predictions but both correct/wrong =====
    diff_both_correct = [r for r in diff_pred if r in both_correct]
    diff_both_wrong = [r for r in diff_pred if r in both_wrong]
    
    if diff_both_correct:
        print(f"\n{'='*80}")
        print(f"DIFFERENT PREDICTIONS, BOTH CORRECT ({len(diff_both_correct)} cases)")
        print(f"{'='*80}")
        for r in diff_both_correct[:10]:
            print(f"  [{r['scene']}] gt={r['gt_id']} | "
                  f"{name_a}: pred={r['pred_id_a']} IoU={r['iou_a']:.4f} | "
                  f"{name_b}: pred={r['pred_id_b']} IoU={r['iou_b']:.4f}")
        if len(diff_both_correct) > 10:
            print(f"  ... and {len(diff_both_correct) - 10} more")
    
    # ===== Per-scene breakdown =====
    print(f"\n{'='*80}")
    print(f"PER-SCENE BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Scene':<20} {'Total':>6} {'Both✓':>7} {'Both✗':>7} {name_a+'✓':>10} {name_b+'✓':>10}")
    print(f"{'-'*60}")
    for scene in sorted(scene_stats.keys()):
        s = scene_stats[scene]
        if s['total'] == 0:
            continue
        print(f"{scene:<20} {s['total']:>6} {s['both_correct']:>7} {s['both_wrong']:>7} "
              f"{s['only_a']:>10} {s['only_b']:>10}")
    
    # ===== Save detailed results to JSON =====
    output = {
        'summary': {
            'iou_threshold': iou_threshold,
            'total': total,
            'both_correct': len(both_correct),
            'both_wrong': len(both_wrong),
            f'only_{name_a}_correct': len(only_a_correct),
            f'only_{name_b}_correct': len(only_b_correct),
            'same_prediction': len(same_pred),
            'diff_prediction': len(diff_pred),
            f'{name_a}_accuracy': acc_a,
            f'{name_b}_accuracy': acc_b,
        },
        'regressions': only_a_correct,
        'improvements': only_b_correct,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
    }
    
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare two prediction versions for 3D Visual Grounding evaluation.")
    parser.add_argument("--dir_a", type=str,
                        default=os.path.join(os.path.dirname(__file__), "/outputs/rex-omni_scanrefer_ov_v1/pred"),
                        help="Path to the first prediction directory")
    parser.add_argument("--dir_b", type=str,
                        default=os.path.join(os.path.dirname(__file__), "/outputs/rex-omni_scanrefer_ov_v0/pred"),
                        help="Path to the second prediction directory")
    parser.add_argument("--name_a", type=str, default="Version A", help="Name for version A")
    parser.add_argument("--name_b", type=str, default="Version B", help="Name for version B")
    args = parser.parse_args()

    dir_a = args.dir_a
    dir_b = args.dir_b
    name_a = args.name_a
    name_b = args.name_b
    
    print("\n" + "="*80)
    print("COMPARISON AT IoU >= 0.25")
    print("="*80)
    result_25 = compare_predictions(dir_a, dir_b, iou_threshold=0.25, name_a=name_a, name_b=name_b)
    
    print("\n\n" + "="*80)
    print("COMPARISON AT IoU >= 0.50")
    print("="*80)
    result_50 = compare_predictions(dir_a, dir_b, iou_threshold=0.50, name_a=name_a, name_b=name_b)
    
    # Save results
    # output_path = os.path.join(os.path.dirname(__file__), "outputs/comparison_results.json")
    # combined = {
    #     'iou_025': result_25,
    #     'iou_050': result_50,
    # }
    
    # with open(output_path, 'w') as f:
    #     json.dump(combined, f, indent=2, default=str)
    
    # print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
