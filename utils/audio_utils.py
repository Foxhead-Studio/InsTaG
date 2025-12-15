import torch
import librosa
from scipy import signal
import numpy as np

# 전체 오디오 피처 중 프레임 인덱스에 해당하는 피처만 잘라서 반환하는 함수
def get_audio_features(features, att_mode, index):
    # features: 오디오 특징 텐서 (예: auds)
    # att_mode: attention 모드 (여기서는 2로 들어옴)
    # index: 현재 프레임 인덱스 (예: idx)
    if att_mode == 0:
        # att_mode가 0이면, index 위치의 특징만 반환
        return features[[index]]
        # 예시: features가 (100, 16, 1)이고 index=5면, features[5]만 반환
    elif att_mode == 1:
        # att_mode가 1이면, index 기준 왼쪽 8개 프레임을 반환
        left = index - 8
        # left: index에서 8을 뺀 값 (예: index=5면 left=-3)
        pad_left = 0
        # pad_left: 왼쪽에 패딩이 필요한 개수 (초기값 0)
        if left < 0:
            # left가 0보다 작으면 (즉, index가 8보다 작으면)
            pad_left = -left
            # pad_left를 -left로 설정 (예: left=-3이면 pad_left=3)
            left = 0
            # left를 0으로 보정 (음수 인덱스 방지)
        auds = features[left:index]
        # features[left:index] 구간의 특징을 auds에 저장
        # 예시: left=0, index=5면 features[0:5] 반환
        if pad_left > 0:
            # pad_left가 0보다 크면 (즉, 실제 데이터가 부족하면)
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
            # 왼쪽에 pad_left만큼 0으로 패딩을 붙여줌
            # 예시: pad_left=3, auds.shape=(2, 16, 1)이면, (3, 16, 1)짜리 0패딩 + (2, 16, 1) auds
        return auds
        # 최종적으로 (8, 16, 1) 형태의 텐서 반환
    elif att_mode == 2:
        # att_mode가 2이면, index 기준 좌우 4개 프레임을 반환
        left = index - 4
        # left: index에서 4를 뺀 값 (예: index=2면 left=-2)
        right = index + 4
        # right: index에서 4를 더한 값 (예: index=2면 right=6)
        pad_left = 0
        # pad_left: 왼쪽 패딩 개수 (초기값 0)
        pad_right = 0
        # pad_right: 오른쪽 패딩 개수 (초기값 0)
        if left < 0:
            # left가 0보다 작으면 (즉, index가 4보다 작으면)
            pad_left = -left
            # pad_left를 -left로 설정 (예: left=-2면 pad_left=2)
            left = 0
            # left를 0으로 보정
        if right > features.shape[0]:
            # right가 features의 길이보다 크면 (즉, 끝을 넘어가면)
            pad_right = right - features.shape[0]
            # pad_right를 right - features.shape[0]로 설정
            right = features.shape[0]
            # right를 features의 길이로 보정
        auds = features[left:right]
        # features[left:right] 구간의 특징을 auds에 저장
        # 예시: left=0, right=6이면 features[0:6] 반환
        if pad_left > 0:
            # pad_left가 0보다 크면 (왼쪽에 데이터가 부족하면)
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
            # auds[:pad_left]와 같은 shape의 0패딩을 왼쪽에 붙임
            # 예시: pad_left=2, auds.shape=(6, 16, 1)이면 (2, 16, 1) 0패딩 + (6, 16, 1) auds
        if pad_right > 0:
            # pad_right가 0보다 크면 (오른쪽에 데이터가 부족하면)
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
            # auds[:pad_right]와 같은 shape의 0패딩을 오른쪽에 붙임
            # 예시: pad_right=1, auds.shape=(8, 16, 1)이면 (8, 16, 1) auds + (1, 16, 1) 0패딩
        return auds
        # 최종적으로 (8, 16, 1) 형태의 텐서 반환
        # 예시: index=2, features 길이=6이면, 왼쪽 2개 0패딩 + features[0:6] + 오른쪽 0패딩(없음)
    else:
        raise NotImplementedError(f'wrong att_mode: {att_mode}')
        # 위 조건에 해당하지 않으면 예외 발생
    
    

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

    return _normalize(S)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel
