import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="calculate the probability result from res/reg caption")
    parser.add_argument("--results_dir",
                        default="", type=str,
                        help="find the caption result")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(f"{args.results_dir}/sampled_captions.json",'r') as f:
        merged_file= json.load(f)
    captions_persam= []
    for idx, caption in enumerate(merged_file):
        if idx%4 == 0 : # 4 from 8 layers
            temp_caption = []
        temp_caption.append(caption)
        if len(temp_caption) == 4:
            sam_total_iou = sum(item['iou'] for item in temp_caption)
            for item in temp_caption:
                item['regular_iou'] = item['iou']/sam_total_iou
            for item in temp_caption:
                item['regular_score'] = item['regular_iou'] * item['sam_score']
            max_dict = max(temp_caption, key=lambda x: x['regular_score'])
            assert len(max_dict) == 8 ,f"{max_dict}_max_dict wrong!!!"
            captions_persam.append(max_dict)

    layer_top_score_set = defaultdict(int)
    for top_caption in captions_persam:
        layer_top_score_set[top_caption['layer']] += top_caption['regular_score'] 

    layer_top_sum = sum(layer_top_score_set.values())
    layer_prob = {key: float(value)/float(layer_top_sum) for key,value in layer_top_score_set.items()}

    sorted_layer_prob = dict(sorted(layer_prob.items()))
    for key, value in sorted_layer_prob.items():
        print('{} layer : {} '.format(key, value))

    
    with open(f"{args.results_dir}/layer_prob.json", "w") as f:
        json.dump(sorted_layer_prob, f, indent=2)

if __name__ == "__main__":
    main()
