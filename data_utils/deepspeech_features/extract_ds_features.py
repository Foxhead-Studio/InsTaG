"""
    Script for extracting DeepSpeech features from audio file.
"""

import os
import argparse
import numpy as np
import pandas as pd
from deepspeech_store import get_deepspeech_model_file
from deepspeech_features import conv_audios_to_deepspeech


def parse_args():
    """
    Create python script parameters.
    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Extract DeepSpeech features from audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input audio file or directory")
    parser.add_argument(
        "--output",
        type=str,
        help="path to output file with DeepSpeech features")
    parser.add_argument(
        "--deepspeech",
        type=str,
        help="path to DeepSpeech 0.1.0 frozen model")
    parser.add_argument(
        "--metainfo",
        type=str,
        help="path to file with meta-information")

    args = parser.parse_args()
    return args


def extract_features(in_audios,
                     out_files,
                     deepspeech_pb_path,
                     metainfo_file_path=None):
    """
    Real extract audio from video file.
    Parameters
    ----------
    in_audios : list of str
        Paths to input audio files.
    out_files : list of str
        Paths to output files with DeepSpeech features.
    deepspeech_pb_path : str
        Path to DeepSpeech 0.1.0 frozen model.
    metainfo_file_path : str, default None
        Path to file with meta-information.
    """
    if metainfo_file_path is None:
        num_frames_info = [None] * len(in_audios)
    else:
        train_df = pd.read_csv(
            metainfo_file_path,
            sep="\t",
            index_col=False,
            dtype={"Id": np.int, "File": np.unicode, "Count": np.int})
        num_frames_info = train_df["Count"].values
        assert (len(num_frames_info) == len(in_audios))

    for i, in_audio in enumerate(in_audios):
        if not out_files[i]:
            file_stem, _ = os.path.splitext(in_audio)
            out_files[i] = file_stem + ".npy"
            #print(out_files[i])
    conv_audios_to_deepspeech(
        audios=in_audios,
        out_files=out_files,
        num_frames_info=num_frames_info,
        deepspeech_pb_path=deepspeech_pb_path)


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_audio = os.path.expanduser(args.input)
    if not os.path.exists(in_audio):
        raise Exception("Input file/directory doesn't exist: {}".format(in_audio))
    deepspeech_pb_path = args.deepspeech
    #add
    deepspeech_pb_path = True
    args.deepspeech = '~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb'
    if deepspeech_pb_path is None:
        deepspeech_pb_path = ""
    if deepspeech_pb_path:
        deepspeech_pb_path = os.path.expanduser(args.deepspeech)
    if not os.path.exists(deepspeech_pb_path):
        deepspeech_pb_path = get_deepspeech_model_file()
    if os.path.isfile(in_audio):
        extract_features(
            in_audios=[in_audio],
            out_files=[args.output],
            deepspeech_pb_path=deepspeech_pb_path,
            metainfo_file_path=args.metainfo)
    else:
        audio_file_paths = []
        for file_name in os.listdir(in_audio):
            if not os.path.isfile(os.path.join(in_audio, file_name)):
                continue
            _, file_ext = os.path.splitext(file_name)
            if file_ext.lower() == ".wav":
                audio_file_path = os.path.join(in_audio, file_name)
                audio_file_paths.append(audio_file_path)
        audio_file_paths = sorted(audio_file_paths)
        out_file_paths = [""] * len(audio_file_paths)
        extract_features(
            in_audios=audio_file_paths,
            out_files=out_file_paths,
            deepspeech_pb_path=deepspeech_pb_path,
            metainfo_file_path=args.metainfo)


if __name__ == "__main__":
    main()

"""
### `extract_ds_features.py` 스크립트 설명

이 스크립트는 오디오 파일에서 **DeepSpeech 특징(features)**을 추출하는 데 사용됩니다.
DeepSpeech 특징은 음성 인식 모델인 DeepSpeech가 음성 데이터를 처리할 때 사용하는 저수준의 음향 특징으로, 주로 음성 합성이나 다른 음성 처리 작업에서 활용됩니다.

#### 1. 스크립트의 목적

주된 목적은 입력으로 제공된 오디오 파일(또는 오디오 파일들이 있는 디렉토리)에서 DeepSpeech 모델을 사용하여 음향 특징을 계산하고, 이를 파일로 저장하는 것입니다.
이렇게 추출된 특징은 원본 오디오를 직접 사용하는 대신, 보다 압축되고 의미 있는 표현으로 음성을 나타낼 수 있게 해줍니다.

#### 2. 필요한 입력 정보

이 스크립트는 실행될 때 몇 가지 정보를 필요로 합니다:

*   **입력 오디오 파일/디렉토리**: 특징을 추출할 대상이 되는 `.wav` 형식의 오디오 파일 경로 또는 여러 오디오 파일이 담긴 디렉토리 경로를 지정해야 합니다.
*   **DeepSpeech 모델 경로 (선택 사항)**: DeepSpeech 특징을 계산하는 데 사용될 사전 학습된 DeepSpeech 모델(`*.pb` 파일)의 경로를 지정할 수 있습니다.
    만약 이 경로가 제공되지 않으면, 스크립트 내에서 기본 DeepSpeech 모델 파일을 자동으로 찾아 사용하려고 시도합니다.
*   **출력 파일 경로 (선택 사항)**: 추출된 특징을 저장할 파일의 경로를 지정할 수 있습니다.
    만약 지정하지 않으면, 입력 오디오 파일과 동일한 이름에 `.npy` 확장자를 붙여 자동으로 생성됩니다.
*   **메타 정보 파일 (선택 사항)**: 오디오 파일에 대한 추가적인 메타 정보(예: 각 오디오의 프레임 수)가 담긴 파일을 제공할 수 있습니다.
    이는 특징 추출 과정에서 특정 처리에 활용될 수 있습니다.

#### 3. 특징 추출 과정

스크립트의 핵심 과정은 DeepSpeech 모델을 활용하여 오디오에서 특징을 추출하는 것입니다:

*   **입력 확인**: 먼저 사용자가 제공한 입력 경로가 유효한 오디오 파일인지, 아니면 오디오 파일들을 담고 있는 디렉토리인지 확인합니다.
*   **모델 준비**: DeepSpeech 모델 파일을 로드하여 특징 추출 준비를 마칩니다.
*   **오디오 처리**:
    *   **단일 오디오 파일**: 하나의 오디오 파일이 입력으로 주어지면, 해당 파일에 대해 DeepSpeech 모델을 실행하여 특징을 직접 추출합니다.
    *   **오디오 디렉토리**: 디렉토리가 주어지면, 그 안에 있는 모든 `.wav` 오디오 파일들을 찾아 목록화합니다.
        그런 다음 각 오디오 파일에 대해 순차적으로 또는 배치 처리 방식으로 DeepSpeech 모델을 실행하여 특징을 추출합니다.
*   **특징 저장**: DeepSpeech 모델로부터 계산된 음향 특징은 각 오디오 파일에 대해 별도의 `.npy` 파일로 저장됩니다.
    이 `.npy` 파일에는 오디오의 시간 흐름에 따른 고차원의 숫자 벡터들이 담겨 있으며, 이 벡터들이 바로 DeepSpeech 특징입니다.

#### 4. 생성되는 출력 정보

이 스크립트를 실행하면 각 오디오 파일에 대해 다음과 같은 `.npy` 파일이 생성됩니다:

*   **DeepSpeech 특징 파일**: `.npy` 확장자를 가진 이진 파일로, 원본 오디오의 음향 특징을 다차원 배열 형태로 저장합니다.
    이 배열의 각 행은 특정 시간 구간의 음향 정보를 나타내는 특징 벡터를 포함합니다.
    오디오 신호는 연속적이지만, 특징을 추출할 때는 보통 20~30ms 정도의 짧은 시간 단위(윈도우)로 나누어 각 구간에서 특징을 계산합니다.
    이 윈도우들은 일반적으로 서로 겹치게 설정되어(예: 10ms씩 이동) 시간적 연속성을 유지합니다.
    따라서 .npy 파일에 저장되는 각 행(row)의 특징 벡터는 이 짧은 오디오 프레임 하나에 해당합니다.
    생성된 .npy 파일의 shape은 일반적으로 (num_audio_frames, feature_dimension) 형태를 가집니다.
    num_audio_frames: 이것은 입력 오디오의 전체 길이와 특징 추출 시 사용된 오디오 프레임의 크기(윈도우 크기) 및 이동 간격(hop length)에 따라 결정되는 오디오 프레임의 총 개수입니다.
    오디오 파일이 길수록 이 값은 커집니다.
    feature_dimension: 이것은 DeepSpeech 모델이 각 오디오 프레임에서 추출하는 특징 벡터의 차원입니다.
    이 값은 DeepSpeech 모델의 설계에 따라 고정된 값입니다.
    예를 들어, 오리지널 DeepSpeech 모델의 경우 각 오디오 프레임에 대해 29차원의 특징 벡터를 생성했습니다.
    예시:
    만약 10초 길이의 오디오 파일에서 DeepSpeech 특징을 추출했고, DeepSpeech 모델이 각 오디오 프레임에서 29차원 특징을 생성하며, 전체 오디오에서 500개의 오디오 프레임이 추출되었다면, .npy 파일의 shape은 (500, 29)가 될 것입니다.
    이 DeepSpeech 특징들은 나중에 비디오 프레임과 시간적으로 정렬되어, 특정 시점의 비디오 프레임에 해당하는 음성 특징을 활용하는 데 사용될 수 있습니다.

이 DeepSpeech 특징은 음성 합성을 위한 인풋으로 사용되거나, 음성 인식 시스템의 특정 계층에서 추가적인 분석을 위한 데이터로 활용될 수 있습니다.
"""