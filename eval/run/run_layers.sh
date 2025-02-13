# !/bin/sh

CAPTION_SCORE_SCRIPTS="eval/run/cyc.sh"
EXP_NAMES=(
    "0-7_four_alldataset_repro_0208"
)

# Start of the range, include
CHOSED_LAYERS_S=(
    0
)
# End of the range, exclude
CHOSED_LAYERS_E=(
    8
)
# Only sample half layers from the range, for saving time
SAMPLE_N_LAYERS=(
    4
)
IDX_NUMS=(
    0 
)
MASTER_PORT=22333


for i in "${!EXP_NAMES[@]}"; do
    EXP_NAME="${EXP_NAMES[$i]}"
    CHOSED_LAYER_S="${CHOSED_LAYERS_S[$i]}"
    CHOSED_LAYER_E="${CHOSED_LAYERS_E[$i]}"
    SAMPLE_N_LAYER="${SAMPLE_N_LAYERS[$i]}"
    IDX_NUM="${IDX_NUMS[$i]}"

    sh $CAPTION_SCORE_SCRIPTS "$EXP_NAME" "$CHOSED_LAYER_S" "$CHOSED_LAYER_E" "$SAMPLE_N_LAYER" "$IDX_NUM"
done
