# Shared launcher for train_rec.py.
# Source this to get the `train` function.
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  : GPU(s) to use (default: 0)
#   CUDA_DEVICE_ORDER     : (default: PCI_BUS_ID)
#   NPROC                 : override number of processes (default: auto from CUDA_VISIBLE_DEVICES)

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Count number of GPUs from CUDA_VISIBLE_DEVICES
_count_gpus() {
    local _devs="${CUDA_VISIBLE_DEVICES}"
    if [[ -z "$_devs" || "$_devs" == "NoDevFiles" ]]; then
        echo 0
    else
        echo $(( $(echo "$_devs" | tr -cd ',' | wc -c) + 1 ))
    fi
}

NPROC="${NPROC:-$(_count_gpus)}"

train() {
    # train <config> [extra args...]
    local _config="$1"; shift
    if [[ "$NPROC" -gt 1 ]]; then
        torchrun --nproc_per_node="${NPROC}" \
            tools/train_rec.py -c "${_config}" "$@"
    else
        python tools/train_rec.py -c "${_config}" "$@"
    fi
}
