# Model config lookup for run_*.sh scripts.
# Source this AFTER setting MODEL.
#
# Sets:
#   CONFIG_BASELINE  - path to the backbone's baseline config (for inference/data prep)
#   CONFIG_PREFIX    - prefix used in configs/rec/ddstr/{PREFIX}_{mode}.yml

case "${MODEL:-svtrv2}" in
    svtrv2)
        CONFIG_BASELINE="configs/rec/svtrv2/svtrv2_rctc.yml"
        CONFIG_PREFIX="svtrv2_rctc"
        ;;
    igtr)
        CONFIG_BASELINE="configs/rec/igtr/svtr_base_ds_igtr.yml"
        CONFIG_PREFIX="igtr"
        ;;
    parseq)
        CONFIG_BASELINE="configs/rec/parseq/svrtv2_parseq.yml"
        CONFIG_PREFIX="svtrv2_parseq"
        ;;
    mdiff4str)
        CONFIG_BASELINE="configs/rec/mdiff4str/svtrv2_mdiffdecoder_base.yml"
        CONFIG_PREFIX="svtrv2_mdiff4str"
        ;;
    *)
        echo "Unknown MODEL: ${MODEL}. Use: svtrv2, igtr, parseq, mdiff4str" >&2
        exit 1
        ;;
esac
