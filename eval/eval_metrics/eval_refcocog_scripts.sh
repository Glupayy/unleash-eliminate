# !/bin/sh

REFCOCOG_SCRIPTS="eval/eval_metrics/eval_meteor_chair_sub.sh"
REFCOCOG_SAVE_PATHS=(
    "0-7_four_alldataset_repro_0208"
)

REFCOCOG_CHOSED_LAYERS=(
    "0,1,2,3,4,5,6,7"
)

for i in "${!REFCOCOG_SAVE_PATHS[@]}"; do
    REFCOCOG_SAVE_PATH="${REFCOCOG_SAVE_PATHS[$i]}"
    REFCOCOG_CHOSED_LAYER="${REFCOCOG_CHOSED_LAYERS[$i]}"

    sh $REFCOCOG_SCRIPTS "$REFCOCOG_SAVE_PATH" "$REFCOCOG_CHOSED_LAYER"
done
