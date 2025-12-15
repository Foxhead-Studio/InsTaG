#!/bin/bash

set -e

EMOTIONS="mark_12s"

for emotion in $EMOTIONS
do
    echo "================================================="
    echo "Processing for emotion: $emotion"
    echo "================================================="

    TARGET_DIR="data/emotion/$emotion"
    VIDEO_FILE="$TARGET_DIR/$emotion.mp4"
    CSV_FILE="$TARGET_DIR/$emotion.csv"
    AUDIO_FILE="$TARGET_DIR/aud.wav"

    echo "STEP 1: Processing video and audio for $VIDEO_FILE"
    python data_utils/process.py "$VIDEO_FILE"

    echo "STEP 2: Extracting OpenFace features"
    /home/white/github/OpenFace/build/bin/FeatureExtraction \
        -f "$VIDEO_FILE" \
        -out_dir "$TARGET_DIR"

    echo "STEP 3: Renaming CSV to au.csv"
    mv "$CSV_FILE" "$TARGET_DIR/au.csv"

    echo "STEP 4: Creating teeth mask"
    export PYTHONPATH=./data_utils/easyportrait
    python ./data_utils/easyportrait/create_teeth_mask.py "$TARGET_DIR"

    echo "STEP 5: Running Sapiens"
    conda run -n sapiens_lite bash ./data_utils/sapiens/run.sh "$TARGET_DIR"

    echo "STEP 6: Extracting DeepSpeech features"
    conda run -n instag python data_utils/deepspeech_features/extract_ds_features.py --input "$AUDIO_FILE"

    echo "STEP 7: Renaming DeepSpeech output"
    mv "$TARGET_DIR/aud.npy" "$TARGET_DIR/aud_ds.npy"

    echo "STEP 8: Extracting Wav2Vec features"
    conda run -n instag python data_utils/wav2vec.py --wav "$AUDIO_FILE" --save_feats

    echo "STEP 9: Extracting HuBERT features"
    conda run -n instag python data_utils/hubert.py --wav "$AUDIO_FILE"

    echo "--- Successfully completed processing for $emotion ---"
    echo ""
done

echo "================================================="
echo "All emotions processed successfully!"
echo "================================================="
