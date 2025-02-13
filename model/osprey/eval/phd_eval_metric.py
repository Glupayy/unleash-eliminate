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
    # coco = COCO(args.annotation_file)
    merged_file_path = f"{args.results_dir}/merged.json"

    if not os.path.exists(merged_file_path):
        # Merge and load the results files
        print("merging splits files..")
        all_results = []
        for result_file in os.listdir(args.results_dir): 
            all_results += json.load(open(f"{args.results_dir}/{result_file}", "r"))
            # all_results.append(one_res)
        # import pdb;pdb.set_trace()
        with open(merged_file_path, 'w') as f:
            # json.dump(all_results, f)
            json.dump(all_results, f)
        with open(merged_file_path, 'r') as f:
            merged_file = json.load(f)

        print(all_results == merged_file)
            
    else:
        print("merged file exists, loading ...")
        with open(merged_file_path,'r') as f:
            merged_file= json.load(f)
        # import pdb;pdb.set_trace()
    label_list = [json.loads(q)['answer'] for q in open(args.annotation_file, 'r')]
    track_list = [json.loads(q)['task']+'_'+json.loads(q)['mode'].split('_')[0]  for q in open(args.annotation_file, 'r')]
    answers = merged_file
    for answer in answers:
        text = answer['answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    ten_track={}
    for pred, label, track in zip(pred_list, label_list, track_list):
        if pred == label :
            if track+"_T" in ten_track:
                ten_track[track+"_T"] += 1
            else: ten_track[track+"_T"] = 1

        if pred != label :
            if track+"_F" in ten_track:
                ten_track[track+"_F"] += 1
            else: ten_track[track+"_F"] = 1
        
    ON = float(ten_track["object_recognition_neutral_T"])/ float(ten_track["object_recognition_neutral_T"]+ten_track["object_recognition_neutral_F"])
    AN = float(ten_track["attribute_recognition_neutral_T"])/ float(ten_track["attribute_recognition_neutral_T"]+ten_track["attribute_recognition_neutral_F"])
    SN = float(ten_track["sentiment_analysis_neutral_T"])/ float(ten_track["sentiment_analysis_neutral_T"]+ten_track["sentiment_analysis_neutral_F"])
    PN = float(ten_track["positional_reasoning_neutral_T"])/ float(ten_track["positional_reasoning_neutral_T"]+ten_track["positional_reasoning_neutral_F"])
    CN = float(ten_track["counting_neutral_T"])/ float(ten_track["counting_neutral_T"]+ten_track["counting_neutral_F"])

    OM = float(ten_track["object_recognition_misleading_T"])/ float(ten_track["object_recognition_misleading_T"]+ten_track["object_recognition_misleading_F"])
    AM = float(ten_track["attribute_recognition_misleading_T"])/ float(ten_track["attribute_recognition_misleading_T"]+ten_track["attribute_recognition_misleading_F"])
    SM = float(ten_track["sentiment_analysis_misleading_T"])/ float(ten_track["sentiment_analysis_misleading_T"]+ten_track["sentiment_analysis_misleading_F"])
    PM = float(ten_track["positional_reasoning_misleading_T"])/ float(ten_track["positional_reasoning_misleading_T"]+ten_track["positional_reasoning_misleading_F"])
    CM = float(ten_track["counting_misleading_T"])/ float(ten_track["counting_misleading_T"]+ten_track["counting_misleading_F"])

    print('ON: {}'.format(ON))
    print('AN: {}'.format(AN))
    print('SN: {}'.format(SN))
    print('PN: {}'.format(PN))
    print('CN: {}'.format(CN))

    print('OM: {}'.format(OM))
    print('AM: {}'.format(AM))
    print('SM: {}'.format(SM))
    print('PM: {}'.format(PM))
    print('CM: {}'.format(CM))

    score = {"ON": ON, 
             "AN": AN,
             "SN": SN,
             "PN": PN,
             "CN": CN,

             "OM": OM, 
             "AM": AM,
             "SM": SM,
             "PM": PM,
             "CM": CM,
             }

    with open(merged_file_path.split('.json')[0]+'metric.json', "w") as f:
        json.dump(score, f, indent=2)


if __name__ == "__main__":
    main()
