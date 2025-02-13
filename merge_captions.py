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
    # parser.add_argument("--caption_persam", default="", type=str, help="find the top max caption result")

    return parser.parse_args()


def main():
    args = parse_args()
    merged_file = {}
    if not os.path.exists(f"{args.results_dir}/caption_persam_merged.json"):
        # Merge and load the results files
        print("merging splits files..")
        all_file = os.listdir(args.results_dir)

        all_file_sub={}
        all_file_sub['caption_all_perlayer'] = [doc for doc in all_file if doc.startswith('caption_all_perlayer')]
        all_file_sub['all_temp_caption'] = [doc for doc in all_file if doc.startswith('all_temp_caption')]
        all_file_sub['caption_persam'] = [doc for doc in all_file if doc.startswith('caption_persam')]
        
        for file_string, file_sub in all_file_sub.items(): 
            all_results = []
            for result_file in file_sub:
                all_results += json.load(open(f"{args.results_dir}/{result_file}", "r"))
            with open(f"{args.results_dir}/{file_string}_merged.json", 'w') as f:
                json.dump(all_results, f)
            with open(f"{args.results_dir}/{file_string}_merged.json", 'r') as f:
                merged_file[file_string] = json.load(f)
            
    

if __name__ == "__main__":
    main()
