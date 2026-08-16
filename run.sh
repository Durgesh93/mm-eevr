#!/bin/bash

pip show papermill >/dev/null 2>&1 || pip install papermill

# ------------------------------------------------------------
# Reduce noisy logs
# ------------------------------------------------------------
export TF_CPP_MIN_LOG_LEVEL=3
export AUTOGRAPH_VERBOSITY=0
export PYTHONWARNINGS="ignore"
export TF_FORCE_GPU_ALLOW_GROWTH=true

# install GNU parallel if missing
if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel not found. Installing..."
    sudo apt-get install -y parallel
fi

# ============================================================
# EDA / PPG runs
# ============================================================

SKIP_EDA_PPG=true   # change to true to skip this block

if [ "$SKIP_EDA_PPG" = "false" ]; then

    NOTEBOOKS=("nb/mlp_eda_ppg.ipynb" "nb/vae_eda_ppg_twostage.ipynb")
    EXPERIMENTS=("SIMPLE" "SMOTE")
    TASKS=("AROUSAL" "VALENCE" "STIMULUS-LABEL")
    MODALITIES=("EDA" "PPG")

    run_job() {
        NOTEBOOK="$1"
        EXPERIMENT="$2"
        MODALITY="$3"
        TASK="$4"

        TMP_NOTEBOOK=$(mktemp /tmp/papermill_XXXXXX.ipynb)

        echo "START pid=$$ | $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY"

        CUDA_VISIBLE_DEVICES="" \
        EXPERIMENT="$EXPERIMENT" TASK="$TASK" MODALITY="$MODALITY" \
        papermill "$NOTEBOOK" "$TMP_NOTEBOOK" --log-output

        rm -f "$TMP_NOTEBOOK"

        echo "DONE pid=$$ | $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY"
    }

    export -f run_job

    parallel --bar --jobs 4 --line-buffer --tag \
        run_job ::: "${NOTEBOOKS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${MODALITIES[@]}" ::: "${TASKS[@]}"

fi

# ============================================================
# TEXT runs
# ============================================================

SKIP_TEXT=true   # change to false to run this block
if [ "$SKIP_TEXT" = "false" ]; then
    NOTEBOOKS=("nb/vae_textdata_twostage.ipynb" "nb/text_classifier_baseline.ipynb")
    EXPERIMENTS=("SIMPLE")
    TASKS=("AROUSAL" "VALENCE" "STIMULUS-LABEL")
    MODALITY="TEXT"

    run_text_job() {
        NOTEBOOK="$1"
        EXPERIMENT="$2"
        TASK="$3"
        MODALITY="$4"
        TMP_NOTEBOOK=$(mktemp /tmp/papermill_text_XXXXXX.ipynb)
        echo "Running $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY"
        EXPERIMENT="$EXPERIMENT" TASK="$TASK" MODALITY="$MODALITY" \
        papermill "$NOTEBOOK" "$TMP_NOTEBOOK" --log-output
        rm -f "$TMP_NOTEBOOK"
        echo "Finished $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY"
    }
    export -f run_text_job
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_text_job ::: "${NOTEBOOKS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "$MODALITY"
fi


# ============================================================
# Common function
# ============================================================
NOTEBOOKS=("nb/mvae.ipynb")
EXPERIMENTS=("SIMPLE")
TASKS=("AROUSAL" "VALENCE" "STIMULUS-LABEL")
MODEL_NAMES=("MoEVAE" "PoEVAE")

run_multimodal_job() {
    NOTEBOOK="$1"
    EXPERIMENT="$2"
    TASK="$3"
    MODALITY_NAME="$4"
    MODEL_NAME="$5"

    TMP_NOTEBOOK=$(mktemp /tmp/papermill_multimodal_XXXXXX.ipynb)

    echo "Running $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY_NAME | $MODEL_NAME"

    EXPERIMENT="$EXPERIMENT" TASK="$TASK" MODALITY="$MODALITY_NAME" MODEL_NAME="$MODEL_NAME" \
    papermill "$NOTEBOOK" "$TMP_NOTEBOOK" --log-output

    rm -f "$TMP_NOTEBOOK"

    echo "Finished $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY_NAME | $MODEL_NAME"
}

export -f run_multimodal_job

SKIP_EDA_PPG=true

if [ "$SKIP_EDA_PPG" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${NOTEBOOKS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "EDA+PPG" ::: "${MODEL_NAMES[@]}"
fi

SKIP_EDA_TEXT=false

if [ "$SKIP_EDA_TEXT" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${NOTEBOOKS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "EDA+TEXT" ::: "${MODEL_NAMES[@]}"
fi

SKIP_PPG_TEXT=false

if [ "$SKIP_PPG_TEXT" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${NOTEBOOKS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "PPG+TEXT" ::: "${MODEL_NAMES[@]}"
fi