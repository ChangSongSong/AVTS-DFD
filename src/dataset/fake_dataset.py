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
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop
from torchaudio.transforms import MelSpectrogram

from dataset.dataset_utils import pil_loader
from dataset.transforms import ToTensorVideo, NormalizeVideo


class FakeDataset(data.Dataset):
    def __init__(
            self,
            mode,
            root,
            input_mode='VA',
            fps=25,
            duration=1,
            grayscale=True,
            img_size=224,
            img_type='face',
            aud_feat='mfcc',
            fake_types=['DF', 'FS', 'F2F', 'NT'],
            use_percentage=100,
    ):
        self.mode = mode
        self.input_mode = input_mode
        self.fps = fps
        self.duration = duration  # seconds
        self.frames_per_clip = int(fps*duration)
        self.grayscale = grayscale
        self.img_size = img_size
        self.img_type = img_type
        self.aud_feat = aud_feat
        self.fake_types = fake_types
        self.fake_types_to_idx = {'real': -1, 'fake': -1}
        for i in range(len(self.fake_types)):
            self.fake_types_to_idx[self.fake_types[i]] = i

        self.video_info = self.get_info(root)
        
        if use_percentage < 100:
            np.random.shuffle(self.video_info)
            self.video_info = self.video_info[:int(len(self.video_info)*use_percentage/100)]

        if self.mode == 'train':
            self.transform = {
                'video': Compose([
                            ToTensorVideo(),
                            RandomCrop((self.img_size, self.img_size)),
                            RandomHorizontalFlip(0.5),
                            NormalizeVideo((0.421,), (0.165,))
                        ]),
                'audio': Compose([
                            MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=512, win_length=512, hop_length=256),
                        ]),
            }
        elif self.mode == 'val' or self.mode == 'test':
            self.transform = {
                'video': Compose([
                            ToTensorVideo(),
                            CenterCrop((self.img_size, self.img_size)),
                            NormalizeVideo((0.421,), (0.165,))
                        ]),
                'audio': Compose([
                            MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=512, win_length=512, hop_length=256),
                        ]),
            }

    def __getitem__(self, index):
        # input info
        vid_path, aud_path, video_type, start_frame = self.video_info[index]

        # video
        if 'V' in self.input_mode:
            vid = self.get_vid(vid_path, start_frame)
        # audio
        if 'A' in self.input_mode:
            aud = self.get_aud(aud_path, start_frame)
        # label
        label = torch.LongTensor([int(video_type != 'real')])

        if self.input_mode == 'V':
            return [vid], label, vid_path
        elif self.input_mode == 'VA':
            return [vid, aud], label, vid_path

    def get_vid(self, vid_path, start_frame):
        # get data paths
        end_frame = start_frame+self.frames_per_clip
        img_paths = [os.path.join(vid_path, f'{i:06d}.jpg') for i in range(start_frame, end_frame, 1)]

        # load imgs
        vid = np.stack([np.array(pil_loader(img_path, self.grayscale)) for img_path in img_paths])

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
                audio, sr = np.load(aud_path)
                audio *= 32768
            else:
                sr, audio = wavfile.read(aud_path)
        elif self.aud_feat == 'melspectrogram':
            audio, sr = torchaudio.load(aud_path)
        else:
            audio, sr = sf.read(aud_path)
        # assert sr == 16000

        # start_time_ms = int((start_frame - 1) * 1000 / self.fps)
        # start_bit = int(start_time_ms * sr / 1000)
        start_bit = int((start_frame - 1) * sr / self.fps)
        end_bit = int(start_bit + sr * self.duration)
        if len(audio.shape) == 2:
            cut_audio = audio[:, start_bit:end_bit]
        elif len(audio.shape) == 1:
            cut_audio = audio[start_bit:end_bit]
        # assert cut_audio.shape[0] == self.duration*sr, (sr, audio.shape, int(start_time_ms/1000), self.duration, cut_audio.shape[0])

        # transform
        if self.aud_feat == 'mfcc':
            aud = python_speech_features.mfcc(cut_audio, samplerate=16000)
            aud = torch.from_numpy(aud).unsqueeze(0)
        elif self.aud_feat == 'melspectrogram':
            aud = self.transform['audio'](cut_audio)

        return aud

    def get_info(self, root):
        dataset_info = []
        splits = json.load(open(os.path.join(root, f'splits/{self.mode}.json')))
        DIR = {
            'real': 'original_sequences/youtube/c23',
            'DF': 'manipulated_sequences/Deepfakes/c23',
            'FS': 'manipulated_sequences/FaceSwap/c23',
            'F2F': 'manipulated_sequences/Face2Face/c23',
            'NT': 'manipulated_sequences/NeuralTextures/c23',
            'FSh': 'manipulated_sequences/FaceShifter/c23',
        }

        label_dict = {
            'real': 'real',
            'DF': 'fake',
            'FS': 'fake',
            'F2F': 'fake',
            'NT': 'fake',
            'FSh': 'fake',
        }
        # label_dict = {
        #     'real': 'real',
        #     'DF': 'DF',
        #     'FS': 'FS',
        #     'F2F': 'F2F',
        #     'NT': 'NT',
        #     'FSh': 'FSh',
        # }

        # follow LipsForensics
        if self.mode == 'train':
            last_frame_dict = {
                'real': 270,
                'DF': 270,
                'FS': 270,
                'F2F': 270,
                'NT': 270,
            }
            skip = 1
        else:
            last_frame_dict = {
                'real': 110,
                'DF': 110,
                'FS': 110,
                'F2F': 110,
                'NT': 110,
                'FSh': 110,
            }
            skip = self.frames_per_clip

        for p in tqdm(splits):
            for i in range(2):
                for t in ['real'] + self.fake_types:
                    if t == 'real':
                        v_name = p[i]
                        a_name = p[i] + '.wav'
                    else:
                        v_name = p[i] + '_' + p[1-i]
                        if t in ['DF', 'FS', 'FSh']:
                            a_name = p[i] + '.wav'
                        elif t in ['F2F', 'NT']:
                            a_name = p[1-i] + '.wav'
                    vid_path = os.path.join(root, DIR[t], self.img_type, v_name)
                    aud_path = os.path.join(root, 'audio_16000', a_name)

                    cond = True
                    if 'V' in self.input_mode:
                        cond &= os.path.isdir(vid_path) and len(os.listdir(vid_path)) > 0
                    if 'A' in self.input_mode:
                        cond &= os.path.isfile(aud_path)

                    if cond:
                        # get vaild frames
                        exist_frames = sorted(list(map(lambda x: int(x[:-4]), os.listdir(vid_path))))
                        first_frame = exist_frames[0]
                        last_frame = last_frame_dict[t]
                        if 'V' in self.input_mode:
                            last_frame = min(last_frame, exist_frames[-1])
                        if 'A' in self.input_mode:
                            sr, aud = wavfile.read(aud_path)
                            aud_len = aud.shape[0]/sr
                            last_frame = min(last_frame, int(aud_len*self.fps-1))
                            
                        for start_frame in range(first_frame, last_frame-self.frames_per_clip+1, skip):
                            end_frame = start_frame+self.frames_per_clip
                            img_paths = [os.path.join(vid_path, f'{i:06d}.jpg') for i in range(
                                start_frame, end_frame, 1)]
                            if all([os.path.isfile(img_path) for img_path in img_paths]):
                                if 'A' in self.input_mode:
                                    dataset_info.append([vid_path, aud_path, label_dict[t], start_frame])
                                else:
                                    dataset_info.append([vid_path, None, label_dict[t], start_frame])

        return dataset_info

    def __len__(self):
        return len(self.video_info)