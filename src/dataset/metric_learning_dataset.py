import os
import torch
import numpy as np
import librosa
import torchaudio
import python_speech_features

from tqdm import tqdm
from torch.utils import data
from scipy.io import wavfile

from dataset.dataset_utils import pil_loader
import time


class MetricLearningDataset(data.Dataset):

    def __init__(
        self,
        root,
        transform,
        mode='train',
        fps=25,
        duration=0.2,
        grayscale=True,
        shift_type='',
        img_type='faces',
        aud_feat='mfcc',
        random_pick_one=False,
    ):
        self.mode = mode
        self.fps = fps
        self.duration = duration  # seconds
        self.grayscale = grayscale
        self.shift_type = shift_type
        self.img_type = img_type
        self.aud_feat = aud_feat

        self.frames_per_clip = int(self.fps * self.duration)

        self.video_info = self.get_info(root)
        self.transform = transform

        self.random_pick_one = random_pick_one

    def __getitem__(self, index):
        st = time.time()
        # input info
        vpath, aud_path = self.video_info[index]

        # video
        if self.shift_type == 'non-overlap':
            skip = self.frames_per_clip
        elif self.shift_type == 'partial':
            skip = 3
        elif self.shift_type == 'all':
            skip = 1
        else:
            raise NotImplementedError

        end_frame = 29 - self.frames_per_clip + 1
        start_frame = np.random.randint(0, skip + end_frame % skip)

        st = time.time()
        all_imgs = torch.from_numpy(
            self.load_img([
                os.path.join(vpath, f'{i:05d}.jpg')
                for i in range(start_frame, end_frame + self.frames_per_clip -
                               1)
            ]))
        vids = []
        for frame in range(start_frame, end_frame, skip):
            if self.random_pick_one:
                if self.mode == 'train':
                    selected_frame = np.random.randint(low=frame,
                                                       high=frame +
                                                       self.frames_per_clip,
                                                       size=1)[0]
                else:
                    selected_frame = frame + self.frames_per_clip // 2
                vid = all_imgs[selected_frame:selected_frame + 1, :, :]
            else:
                vid = all_imgs[frame:frame + self.frames_per_clip, :, :]
            if self.grayscale:
                vid = vid.unsqueeze(-1)
            if self.transform is not None and 'video' in self.transform:
                vid = self.transform['video'](vid)
            vids.append(vid)
        vids = torch.stack(vids, 0)

        st = time.time()
        # audio
        auds = []
        sr, audio = wavfile.read(aud_path)
        # audio, sr = torchaudio.load(aud_path)
        # audio, sr = librosa.load(aud_path, sr=16000)
        # audio = np.load(aud_path)
        sr = 16000
        for frame in range(start_frame, end_frame, skip):
            start_bit = int(frame * sr / self.fps)
            end_bit = int((frame + self.frames_per_clip) * sr / self.fps)
            cut_audio = audio[start_bit:end_bit]

            if self.aud_feat == 'mfcc':
                aud = python_speech_features.mfcc(cut_audio, samplerate=16000)
                # aud = librosa.feature.mfcc(y=cut_audio, sr=16000, n_mels=13)
                aud = torch.from_numpy(aud).unsqueeze(0)
            elif self.aud_feat == 'melspectrogram':
                if self.transform is not None and 'audio' in self.transform:
                    aud = self.transform['audio'](cut_audio)
            else:
                raise NotImplementedError

            auds.append(aud)
        auds = torch.stack(auds, 0)

        # label
        label = torch.arange(len(vids)).repeat(2)

        return vids, auds, label, vpath

    def load_img(self, img_paths):
        vid = np.stack([
            np.array(pil_loader(img_path, self.grayscale))
            for img_path in img_paths
        ])

        return vid

    def get_info(self, root):
        print(f'Getting {self.mode} dataset...')
        img_root = os.path.join(root, self.img_type)
        aud_root = os.path.join(root, 'audios')
        # aud_root = os.path.join(root, 'audios_npy')

        video_info = []
        for c in tqdm(os.listdir(img_root)):
            if not os.path.exists(os.path.join(img_root, c, self.mode)):
                continue

            # run a smaller experiment
            d_ratio = 1.0
            ppl = os.listdir(os.path.join(img_root, c, self.mode))
            ppl = np.random.choice(ppl, int(len(ppl) * d_ratio)).tolist()
            for person in ppl:
                img_dir = os.path.join(img_root, c, self.mode, person)
                aud_path = os.path.join(aud_root, c, self.mode,
                                        person + '.wav')
                # aud_path = os.path.join(aud_root, c, self.mode, person+'.npy')
                if len(os.listdir(img_dir)) != 29 or not os.path.exists(
                        aud_path):
                    continue
                video_info.append([img_dir, aud_path])
        return video_info

    def __len__(self):
        return len(self.video_info)
