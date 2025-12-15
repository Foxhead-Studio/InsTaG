from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch

print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    else:
        print("skip the latest ", input_values.shape[1])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

import soundfile as sf
import numpy as np
import torch
from argparse import ArgumentParser
import librosa

parser = ArgumentParser()
parser.add_argument('--wav', type=str, help='')
args = parser.parse_args()

wav_name = args.wav

speech_16k, sr = librosa.load(wav_name, sr=16000)
# speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
# print("SR: {} to {}".format(sr, 16000))
# print(speech.shape, speech_16k.shape)

hubert_hidden = get_hubert_from_16k_speech(speech_16k)
hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
np.save(wav_name.replace('.wav', '_hu.npy'), hubert_hidden.detach().numpy())
print(hubert_hidden.detach().numpy().shape)

"""
### `hubert.py` 스크립트 설명

이 스크립트는 **HuBERT(Hidden-Unit Bidirectional Encoder Representations from Transformers)** 모델을 사용하여 오디오 파일에서 **음성 특징(features)**을 추출하는 데 특화되어 있습니다.
HuBERT 특징은 음성 신호의 숨겨진 의미론적(semantic) 또는 음향학적(acoustic) 정보를 포착하도록 훈련된 고차원 표현으로, 주로 음성 인식, 음성 합성, 화자 인식 등 다양한 음성 관련 작업에서 효과적인 입력으로 사용됩니다.

#### 1. 스크립트의 목적

주된 목적은 16kHz 샘플링 레이트의 `.wav` 오디오 파일을 입력으로 받아, 사전 학습된 HuBERT 모델을 통과시켜 해당 오디오의 고차원 음성 특징을 추출하고 이를 파일로 저장하는 것입니다.
이 특징들은 원본 오디오 파형보다 훨씬 추상적이고 정보 밀도가 높은 표현으로, 음성 데이터의 핵심 정보를 담고 있습니다.

#### 2. 사용 모델 및 초기화

스크립트는 Hugging Face `transformers` 라이브러리를 통해 **사전 학습된 HuBERT 모델**을 사용합니다.
특히 "facebook/hubert-large-ls960-ft"와 같은 모델 체크포인트를 불러와 `Wav2Vec2Processor`와 `HubertModel`을 초기화합니다.
`Wav2Vec2Processor`는 오디오 데이터를 모델이 이해할 수 있는 형태로 전처리하는 역할을 하며, `HubertModel`은 실제 특징을 추출하는 딥러닝 모델입니다.
모델은 GPU(cuda:0)로 로드되어 효율적인 연산을 수행합니다.

#### 3. HuBERT 특징 추출 과정

특징 추출은 주로 `get_hubert_from_16k_speech` 함수에서 이루어지며, 다음과 같은 단계를 거칩니다:

*   **오디오 로드 및 전처리**: 입력으로 16kHz 샘플링 레이트의 오디오(`speech_16k`)를 받습니다.
    오디오가 여러 채널일 경우 첫 번째 채널만 사용하며, `Wav2Vec2Processor`를 통해 모델 입력에 맞는 형태로 변환됩니다.
*   **장시간 오디오 처리**: HuBERT 모델은 긴 오디오 시퀀스를 한 번에 처리하기에는 메모리 제약이 있습니다.
    이를 해결하기 위해 스크립트는 긴 오디오를 여러 개의 작은 "클립"으로 나누어 순차적으로 처리합니다.
    각 클립은 HuBERT 모델의 내부 구조(특정 커널 크기와 스트라이드를 가진 CNN)를 고려하여 효율적으로 처리될 수 있도록 길이가 조정됩니다.
*   **특징 계산**: 각 오디오 클립은 HuBERT 모델의 `forward` 함수를 통해 처리됩니다.
    이 과정에서 모델의 마지막 은닉 상태(last hidden state)가 추출되는데, 이것이 바로 HuBERT 특징입니다.
    이 특징들은 보통 `(배치 크기, 시간 단계 수, 특징 차원)` 형태를 가집니다.
*   **특징 결합 및 정제**: 여러 클립에서 추출된 특징들은 하나로 합쳐집니다.
    이어서, 모델의 출력 시간 단계 수와 실제 오디오 길이에 기반한 예상 시간 단계 수가 정확히 일치하도록 길이를 조정하거나 패딩(padding)을 추가하는 후처리 과정이 적용됩니다.

#### 4. 생성되는 출력 정보

스크립트는 최종적으로 다음과 같은 `.npy` 파일을 생성합니다:

*   **HuBERT 특징 파일**: 입력 `.wav` 파일과 동일한 이름에 `_hu.npy` 확장자를 붙인 파일(예: `audio.wav` -> `audio_hu.npy`)이 생성됩니다.
*   **데이터 형식**: 이 파일에는 추출된 HuBERT 특징이 NumPy 배열 형태로 저장됩니다.
*   **Shape**: 일반적으로 `(num_audio_frames, feature_dimension)` 형태를 가지며, 여기서 `feature_dimension`은 HuBERT 모델의 크기에 따라 다르지만, 이 스크립트에서 사용하는 `hubert-large` 모델의 경우 1024차원입니다.
    또한, 최종적으로 `make_even_first_dim` 함수를 통해 첫 번째 차원(오디오 프레임 수)이 짝수가 되도록 조정하고, `(num_audio_frames / 2, 2, 1024)`와 같은 형태로 재구성될 수 있습니다.
    이는 아마도 후속 모델의 특정 입력 요구 사항을 맞추기 위함일 것입니다.

이 HuBERT 특징은 DeepSpeech나 Wav2Vec2 특징보다 더 추상적이고 풍부한 음성 정보를 담고 있어, 복잡한 음성 표현 학습이나 텍스트-음성 정렬과 같은 고급 음성 처리 작업에 활용될 수 있습니다.
"""