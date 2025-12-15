#!/bin/bash

# set -e : 에러 발생 시 즉시 종료
# set -x : 실행되는 명령어를 터미널에 출력 (디버깅 용도)
# chmod +x scripts/process_MEAD.sh
# ./scripts/process_MEAD.sh
set -e
set -x

# 배열 선언
EMOTIONS=("front" "left_30")

# [중요] 배열 전체를 돌기 위해 "${EMOTIONS[@]}" 사용
for emotion in "${EMOTIONS[@]}"
do
    # set -x 때문에 echo가 중복 출력될 수 있으니, 구분을 위해 set +x로 잠시 끄기도 함 (선택사항)
    set +x 
    echo ""
    echo "================================================="
    echo "Processing for emotion: $emotion"
    echo "================================================="
    set -x # 다시 명령어 출력 켜기

    TARGET_DIR="data/MEAD/027/$emotion"
    VIDEO_FILE="$TARGET_DIR/$emotion.mp4"
    CSV_FILE="$TARGET_DIR/$emotion.csv"
    AUDIO_FILE="$TARGET_DIR/aud.wav"

    # STEP 1
    echo "STEP 1: Processing video and audio for $VIDEO_FILE"
    python data_utils/process.py "$VIDEO_FILE"

    # STEP 2
    echo "STEP 2: Extracting OpenFace features"
    /home/white/github/OpenFace/build/bin/FeatureExtraction \
        -f "$VIDEO_FILE" \
        -out_dir "$TARGET_DIR"

    # STEP 3
    echo "STEP 3: Renaming CSV to au.csv"
    mv "$CSV_FILE" "$TARGET_DIR/au.csv"

    # STEP 4
    echo "STEP 4: Creating teeth mask"
    export PYTHONPATH=./data_utils/easyportrait
    python ./data_utils/easyportrait/create_teeth_mask.py "$TARGET_DIR"

    # STEP 5
    echo "STEP 5: Running Sapiens"
    conda run -n sapiens_lite bash ./data_utils/sapiens/run.sh "$TARGET_DIR"

    # STEP 6
    echo "STEP 6: Extracting DeepSpeech features"
    conda run -n instag python data_utils/deepspeech_features/extract_ds_features.py --input "$AUDIO_FILE"

    # STEP 7
    echo "STEP 7: Renaming DeepSpeech output"
    mv "$TARGET_DIR/aud.npy" "$TARGET_DIR/aud_ds.npy"

    # STEP 8
    echo "STEP 8: Extracting Wav2Vec features"
    conda run -n instag python data_utils/wav2vec.py --wav "$AUDIO_FILE" --save_feats

    # STEP 9
    echo "STEP 9: Extracting HuBERT features"
    conda run -n instag python data_utils/hubert.py --wav "$AUDIO_FILE"

    set +x
    echo "--- Successfully completed processing for $emotion ---"
    echo ""
    set -x
done

set +x
echo "================================================="
echo "All emotions processed successfully!"
echo "================================================="
