#!/usr/bin/env bash
set -euo pipefail

# ===== Pretrain IDs =====
for ID in Jae-in macron may Obama1 Shaheen; do
  echo ">>> Pretrain: $ID"
  python data_utils/process.py data/pretrain/$ID/$ID.mp4
  python data_utils/split.py   data/pretrain/$ID/$ID.mp4
done

# ===== Test/Adapt IDs =====
for ID in Lieu Macron May Obama Obama2; do
  echo ">>> Test/Adapt: $ID"
  python data_utils/process.py data/$ID/$ID.mp4
  python data_utils/split.py   data/$ID/$ID.mp4
done
