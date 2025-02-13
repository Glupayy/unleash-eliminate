import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")

    parser.add_argument("--annotation_file",
                        default="data/RefCoco_Reg/mdetr_annotations/finetune_refcocog_val_captions.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load the annotation file
    coco = COCO(args.annotation_file)
    merged_file_path = f"{args.results_dir}/caption_persam_merged.json"
    print("merged_file_path:",merged_file_path)

    if not os.path.exists(merged_file_path):
        # # Merge and load the results files
        # print("merging splits files..")
        # all_results = []
        # for result_file in os.listdir(args.results_dir): 
        #     all_results += json.load(open(f"{args.results_dir}/{result_file}", "r"))
        #     # all_results.append(one_res)
        # # import pdb;pdb.set_trace()
        # with open(merged_file_path, 'w') as f:
        #     json.dump(all_results, f)
        print("no merged file, try again!")
        return 0
    else:
        print("merged file exists, loading ...")
        with open(merged_file_path,'r') as f:
            merged_file= json.load(f)
        # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    
    coco_result = coco.loadRes(merged_file_path)
    # import pdb;pdb.set_trace()
    # print(coco_result)
    # Create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate results
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # Print and save the output evaluation scores
    output_file_path = f"{args.results_dir}/metrics.txt"
    f = open(output_file_path, 'w')
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        f.write(f"{metric}: {score:.3f}\n")
    f.close()


if __name__ == "__main__":
    main()
