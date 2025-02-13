## The implementation of ***Unleashing Region Understanding in Intermediate Layers for MLLM-based Referring Expression Generation***.
TLDR: We propose a simple and effective method to leverage the intermediate layers of MLLMs for increasing descriptive information and mitigating hallucinations.

## Installation
This codebase is built upon [GLAMM](https://github.com/mbzuai-oryx/groundingLMM) and [Osprey](https://github.com/CircleRadon/Osprey). To set up the environment properly, please follow the instructions provided in their respective repositories.

To simplify the setup process, we offer the [conda environment](https://drive.google.com/file/d/17fWE1n37f7nBJdwPPiLApBP_dmhnFywm/view?usp=drive_link). After installing this environment, you can install the editable packages as follows:
```bash
    git clone git@github.com:Glupayy/unleash-eliminate.git
    cd transformers_cyc
    pip install -e .
    cd ../mmcv
    MMCV_WITH_OPS=1 pip install -e .
```
Prepare the [RefCOCOg](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/datasets.md) and [PHD](https://github.com/jiazhen-code/PhD) dataset.

Download the models of [GLAMM](https://huggingface.co/MBZUAI/GLaMM-RefSeg) and [Osprey](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main). 

Change these model/data arguments in [main.py](main.py):
`version_res, version_reg, vision_pretrained, vision_tower, refcoco_image, anno_path`


## A. Cycle-consistency-based Quality Ranking
First unleash the latent information of candidate layers by contrastive decoding. Then use the model of dual-task (RES) to score the intermediate layer outputs of the contrastive decoding. In this case, the highest-scoring output is taken as the model output. 

Example output: 
0-7 as candidates: [scores_0_7.json](files/scores_0_7.json); 8-15 as candidates: [scores_8_15.json](files/scores_8_15.json)

Running scripts:
```bash
sh eval/run/run_layers.sh
#evaluation: METEOR & CHAIRs
sh huanggb_check_three_metrics/eval_refcocog_scripts_huanggb.sh 
```


## B. Probing-based Hybrid Layer Importance Measurement
First sample the subset randomly. [Our subset](output/0-7_four_alldataset_repro_0208/sampled_captions.json).
Then calculate the **layer importance prior**:
```bash
python calculate_prob.py --results_dir output/0-7_four_alldataset_repro_0208
# define the PHD_LAYERS_PROB as importance prior in the script
# evaluate hallucinations benchmark with layer importance prior
sh eval/run_with_prob/eval_phd.sh
```



