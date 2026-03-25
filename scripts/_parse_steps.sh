# Shared step parser for run_*.sh scripts.
# Source this AFTER setting STEP_ALIASES (associative array) and DEFAULT_STEPS.
#
# Supports: --steps=4          (just step 4)
#           --steps=123        (steps 1,2,3)
#           --steps=CLD        (alias, expanded per script)
#           --steps=123,CLD    (mixed)
#           --steps=all        (everything)
#
# Sets: STEPS (expanded string of step numbers), PASS_ARGS (remaining args)

_RAW_STEPS="${DEFAULT_STEPS}"
PASS_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --steps=* ]]; then
        _RAW_STEPS="${arg#--steps=}"
    else
        PASS_ARGS+=("$arg")
    fi
done

# Expand aliases: split on comma, replace aliases with their expansion
STEPS=""
IFS=',' read -ra _PARTS <<< "$_RAW_STEPS"
for _part in "${_PARTS[@]}"; do
    if [[ -n "${STEP_ALIASES[$_part]+x}" ]]; then
        STEPS+="${STEP_ALIASES[$_part]}"
    else
        STEPS+="$_part"
    fi
done

run_step() { [[ "$STEPS" == "all" ]] || [[ "$STEPS" == *"$1"* ]]; }
elapsed() {
    local _s=$(( SECONDS - _T0 ))
    local _h=$(( _s / 3600 )) _m=$(( _s % 3600 / 60 )) _sec=$(( _s % 60 ))
    printf -- "--- Step %s done in %dh %dm %ds ---\n" "$1" "$_h" "$_m" "$_sec"
}
