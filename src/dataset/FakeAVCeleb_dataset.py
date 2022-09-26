import os
import json
import torch
import torchaudio
import python_speech_features

import numpy as np
import soundfile as sf

from tqdm import tqdm
from torch.utils import data
from scipy.io import wavfile
from torchvision.transforms import Compose, CenterCrop, RandomCrop, RandomHorizontalFlip
from torchaudio.transforms import MelSpectrogram

from dataset.dataset_utils import pil_loader
from dataset.transforms import ToTensorVideo, NormalizeVideo


class FakeAVCelebDataset(data.Dataset):

    def __init__(
        self,
        mode,
        root='/scratch3/users/cssung/datasets/FakeAVCeleb',
        fps=25,
        duration=1,
        grayscale=True,
        img_size=224,
        img_type='face',
        aud_feat='mfcc',
        use_percentage=100,
        pick_one=False,
    ):
        self.mode = mode
        self.fps = fps
        self.duration = duration  # seconds
        self.frames_per_clip = int(fps * duration)
        self.grayscale = grayscale
        self.img_size = img_size
        self.img_type = img_type
        self.aud_feat = aud_feat

        self.pick_one = pick_one

        self.video_info = self.get_info(root)

        if use_percentage < 100:
            np.random.shuffle(self.video_info)
            self.video_info = self.video_info[:int(
                len(self.video_info) * use_percentage / 100)]

        if self.mode == 'train':
            self.transform = {
                'video':
                Compose([
                    ToTensorVideo(),
                    RandomCrop((self.img_size, self.img_size)),
                    RandomHorizontalFlip(0.5),
                    NormalizeVideo((0.421, ), (0.165, ))
                ]),
                'audio':
                Compose([
                    # Gain(min_gain=-10, max_gain=10),
                    MelSpectrogram(sample_rate=16000,
                                   n_mels=64,
                                   n_fft=512,
                                   win_length=512,
                                   hop_length=256),
                ]),
            }
        elif self.mode == 'val' or self.mode == 'test':
            self.transform = {
                'video':
                Compose([
                    ToTensorVideo(),
                    CenterCrop((self.img_size, self.img_size)),
                    NormalizeVideo((0.421, ), (0.165, ))
                ]),
                'audio':
                Compose([
                    MelSpectrogram(sample_rate=16000,
                                   n_mels=64,
                                   n_fft=512,
                                   win_length=512,
                                   hop_length=256),
                ]),
            }

    def __getitem__(self, index):
        # input info
        vid_path, aud_path, video_label, start_frame = self.video_info[index]

        # video
        vid = self.get_vid(vid_path, start_frame)
        if self.pick_one:
            if self.mode == 'train':
                selected_frame = np.random.randint(self.frames_per_clip,
                                                   size=1)[0]
            else:
                selected_frame = self.frames_per_clip // 2
            vid = vid[:, selected_frame:selected_frame + 1, :, :]
        # audio
        aud = self.get_aud(aud_path, start_frame)
        # label
        label = torch.LongTensor([video_label])

        return [vid, aud], label, vid_path

    def get_vid(self, vid_path, start_frame):
        # get data paths
        end_frame = start_frame + self.frames_per_clip
        img_paths = [
            os.path.join(vid_path, f'{i:06d}.jpg')
            for i in range(start_frame, end_frame, 1)
        ]

        # load imgs
        vid = np.stack([
            np.array(pil_loader(img_path, self.grayscale))
            for img_path in img_paths
        ])

        vid = torch.from_numpy(vid)
        if self.grayscale:
            vid = vid.unsqueeze(-1)
        vid = self.transform['video'](vid)

        return vid

    def get_aud(self, aud_path, start_frame, sr=16000):
        # recalculate end_time due to int precision error, may cause dimension error
        # read ori audio
        if self.aud_feat == 'mfcc':
            if aud_path.endswith('npy'):
                audio = np.load(aud_path)
                audio *= 32768
            else:
                sr, audio = wavfile.read(aud_path)
        elif self.aud_feat == 'melspectrogram':
            audio, sr = torchaudio.load(aud_path)
        else:
            audio, sr = sf.read(aud_path)

        start_bit = int((start_frame - 1) * sr / self.fps)
        end_bit = int(start_bit + sr * self.duration)

        if len(audio.shape) == 2:
            cut_audio = audio[:, start_bit:end_bit]
        elif len(audio.shape) == 1:
            cut_audio = audio[start_bit:end_bit]

        # transform
        if self.aud_feat == 'mfcc':
            aud = python_speech_features.mfcc(cut_audio, samplerate=16000)
            aud = torch.from_numpy(aud).unsqueeze(0)
        elif self.aud_feat == 'melspectrogram':
            aud = self.transform['audio'](cut_audio)

        return aud

    def get_info(self, root):
        dataset_info = []

        splits = json.load(open(os.path.join(root,
                                             f'splits/{self.mode}.json')))
        races = ['AFRICANAMERICAN', 'American', 'Asian', 'European', 'Indian']
        genders = ['MEN', 'WOMEN']
        video_dir_names = [
            'REAL_A', 'FAKE_RTVC_B',
            ['FAKE_FACESWAP_C', 'FAKE_FSGAN_C', 'FAKE_W2L_C'],
            ['FAKE_FACESWAP_D_W2L', 'FAKE_FSGAN_D_W2L', 'org_FAKE_W2L_D']
        ]
        labels = [0, 1, 1, 1]

        if self.mode == 'train':
            roots = [
                os.path.join(root, self.img_type, 'TRAIN',
                             'ARRANGED_RAW_VIDEO'),
                os.path.join(root, 'audio_npy', 'TRAIN', 'ARRANGED_RAW_VIDEO')
            ]
            skip = 1
        elif self.mode == 'val':
            roots = [
                os.path.join(root, self.img_type, 'TRAIN',
                             'ARRANGED_RAW_VIDEO'),
                os.path.join(root, 'audio_npy', 'TRAIN', 'ARRANGED_RAW_VIDEO')
            ]
            skip = 1
        elif self.mode == 'test':
            roots = [
                os.path.join(root, self.img_type, 'TEST',
                             'ARRANGED_RAW_VIDEO'),
                os.path.join(root, 'audio_npy', 'TEST', 'ARRANGED_RAW_VIDEO')
            ]
            skip = 1

        for race in tqdm(races):
            for gender in genders:
                vids = splits[race][gender]

                data_root_a = [[
                    os.path.join(roots[0], video_dir_names[0]),
                    os.path.join(roots[1], video_dir_names[0])
                ]]
                data_root_b = [[
                    os.path.join(roots[0], video_dir_names[1]),
                    os.path.join(roots[1], video_dir_names[1])
                ]]
                data_root_c = [[
                    os.path.join(roots[0], n),
                    os.path.join(roots[1], n)
                ] for n in video_dir_names[2]]
                data_root_d = [[
                    os.path.join(roots[0], n),
                    os.path.join(roots[1], n)
                ] for n in video_dir_names[3]]

                for i, data_root in enumerate(
                    [data_root_a, data_root_b, data_root_c, data_root_d]):
                    label = labels[i]
                    for img_root, aud_root in data_root:
                        curr_img_dir = os.path.join(img_root, race, gender)
                        curr_aud_dir = os.path.join(aud_root, race, gender)

                        for vid in vids:
                            if not os.path.isdir(
                                    os.path.join(curr_img_dir, vid)):
                                continue
                            for correspoing_id in os.listdir(
                                    os.path.join(curr_img_dir, vid)):
                                vid_path = os.path.join(
                                    curr_img_dir, vid, correspoing_id)
                                aud_path = os.path.join(
                                    curr_aud_dir, vid, correspoing_id + '.npy')

                                if os.path.isdir(vid_path) and os.path.isfile(
                                        aud_path):
                                    # get vaild frames
                                    exist_frames = sorted(
                                        list(
                                            map(lambda x: int(x[:-4]),
                                                os.listdir(vid_path))))
                                    first_frame = exist_frames[0]
                                    aud = np.load(aud_path)
                                    sr = 16000
                                    aud_len = aud.shape[0] / sr
                                    last_frame = min(
                                        exist_frames[-1],
                                        int(aud_len * self.fps - 1))

                                    for start_frame in range(
                                            first_frame, last_frame -
                                            self.frames_per_clip + 1, skip):
                                        end_frame = start_frame + self.frames_per_clip
                                        img_paths = [
                                            os.path.join(
                                                vid_path, f'{i:06d}.jpg')
                                            for i in range(
                                                start_frame, end_frame, 1)
                                        ]
                                        if all([
                                                os.path.isfile(img_path)
                                                for img_path in img_paths
                                        ]):
                                            dataset_info.append([
                                                vid_path, aud_path, label,
                                                start_frame
                                            ])

        return dataset_info

    def __len__(self):
        return len(self.video_info)
