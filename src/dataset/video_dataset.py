import os
import torch
import python_speech_features
import numpy as np
import pandas as pd
import soundfile as sf

from torch.utils import data
from scipy.io import wavfile

from dataset.dataset_utils import pil_loader


class VideoDataset(data.Dataset):

    def __init__(
        self,
        root,
        transform,
        mode='train',
        input_mode='VA',
        fps=25,
        duration=1,
        sample_rate=1,
        grayscale=True,
        augmented_type='',
        shift_type='',
        aud_feat='wav',
    ):
        self.mode = mode
        self.input_mode = input_mode
        self.fps = fps
        self.duration = duration  # seconds
        self.sample_rate = sample_rate
        self.grayscale = grayscale
        self.augmented_type = augmented_type
        self.shift_type = shift_type
        self.aud_feat = aud_feat

        self.video_info = pd.read_csv(root, header=None)
        self.transform = transform

    def __getitem__(self, index):
        # input info
        if self.mode == 'train':
            vpath, audiopath, label, ok_frames, video_fps = self.video_info.iloc[
                index]
            ok_frames = list(eval(ok_frames))  # turn string to list
            start_frame = np.random.choice(ok_frames, 1)[0]
        elif self.mode == 'val' or self.mode == 'test':
            vpath, audiopath, label, start_frame, video_fps = self.video_info.iloc[
                index]

        if self.augmented_type in ['sync']:
            if self.mode == 'train':
                self.augmented_signal = np.random.choice([0, 1], 1)[0]
                shift_sec = None
            else:
                self.augmented_signal = int(label != 0)
                shift_sec = label
        else:
            shift_sec = 0

        # video
        if 'V' in self.input_mode:
            img_paths = self.choose_frames(vpath, start_frame, video_fps)
            vid = self.load_img(img_paths)
            vid = torch.from_numpy(vid).unsqueeze(-1)
            if self.transform is not None and 'video' in self.transform:
                vid = self.transform['video'](vid)

        # audio
        if 'A' in self.input_mode:
            if 'FaceForensics' in audiopath or 'DeeperForensics' in audiopath or 'DFDC' in audiopath or 'FakeAVCeleb_v1_2' in audiopath or 'FakeAVCeleb' in audiopath:
                start_time = (start_frame - 1) / video_fps  # for FF++
            elif 'LRW' in audiopath:
                start_time = start_frame / video_fps  # for LRW
            else:
                raise ValueError
            aud = self.crop_audio(audiopath, start_time, shift_sec=shift_sec)
            if self.aud_feat == 'mfcc':
                # MFCC
                aud = python_speech_features.mfcc(aud, samplerate=16000)
            elif self.aud_feat == 'wav':
                if self.transform is not None and 'audio' in self.transform:
                    aud = self.transform['audio'](aud)
            aud = torch.from_numpy(aud).unsqueeze(0)

        # label
        if self.augmented_type in ['sync']:
            label = torch.FloatTensor([int(self.augmented_signal)])
        else:
            label = torch.FloatTensor([label])

        if self.input_mode == 'VA':
            data = [[vid, aud], label, vpath]
        elif self.input_mode == 'V':
            data = [[vid], label, vpath]
        elif self.input_mode == 'A':
            data = [[aud], label, vpath]

        return data

    def choose_frames(self, vpath, start_frame, video_fps):
        end_frame = start_frame + int(video_fps * self.duration)
        img_paths = [
            os.path.join(vpath, f'{i:05d}.jpg')
            for i in range(start_frame, end_frame, self.sample_rate)
        ]
        img_paths = self.check_img_paths_valid(img_paths)

        return img_paths

    def load_img(self, img_paths):
        vid = np.stack([
            np.array(pil_loader(img_path, self.grayscale))
            for img_path in img_paths
        ])
        if self.augmented_type == 'flip' and self.augmented_signal:
            start_flip_idx = np.random.randint(
                0, self.fps * self.duration - self.flip_num + 1)
            end_flip_idx = start_flip_idx + self.flip_num
            vid = np.concatenate((vid[:start_flip_idx],
                                  np.flip(vid[start_flip_idx:end_flip_idx],
                                          2).copy(), vid[end_flip_idx:]), 0)

        return vid

    def check_img_paths_valid(self, img_paths):
        new_pth = []
        for i in range(len(img_paths)):
            if not os.path.isfile(img_paths[i]):
                new_pth.append(self.find_most_relevent(img_paths[i]))
            else:
                new_pth.append(img_paths[i])
        return new_pth

    def crop_audio(self, audiopath, start_time, sr=16000, shift_sec=0):
        audio_type = audiopath.split('.')[-1]
        if audio_type == 'npz':
            audio = np.load(audiopath)['data']
            if self.aud_feat == 'mfcc':
                audio *= 32768
        elif audio_type == 'npy':
            audio = np.load(audiopath)
            if self.aud_feat == 'mfcc':
                audio *= 32768
        elif audio_type == 'wav':
            if self.aud_feat == 'mfcc':
                sr, audio = wavfile.read(audiopath)
            else:
                audio, sr = sf.read(audiopath)
            assert sr == 16000

        # recalculate end_time due to int precision error, may cause dimension error
        if self.augmented_type == 'sync' and self.augmented_signal:
            audio_length = audio.shape[0] / sr
            if self.mode == 'train':
                if 'FaceForensics' in audiopath:
                    aud_len_ms = int(audio_length * 1000)  # for FF++
                elif 'LRW' in audiopath:
                    aud_len_ms = int(min(audio_length * 1000, 1120))  # for LRW

                # decide shift range
                start_time_ms = int(start_time * 1000)
                duration_ms = int(self.duration * 1000)
                frame_ms = int(1000 / self.fps)
                if self.shift_type == 'non-overlap':
                    shift_choices = [
                        s / 1000
                        for s in range(0, start_time_ms - duration_ms +
                                       1, frame_ms)
                    ] + [
                        s / 1000 for s in range(start_time_ms +
                                                duration_ms, aud_len_ms -
                                                duration_ms + 1, frame_ms)
                    ]
                elif self.shift_type == 'overlap':
                    shift_choices = [
                        s / 1000
                        for s in range(start_time_ms - duration_ms +
                                       frame_ms, start_time_ms +
                                       duration_ms, frame_ms)
                        if s != start_time_ms
                    ]
                elif self.shift_type == 'all':
                    shift_choices = [
                        s / 1000 for s in range(0, aud_len_ms - duration_ms +
                                                1, frame_ms)
                        if s != start_time_ms
                    ]
                else:
                    raise NotImplementedError

                shifted_time = np.random.choice(shift_choices)
            else:
                # shift back
                if start_time - shift_sec < 0:
                    shifted_time = start_time + shift_sec
                # shift forward
                else:
                    shifted_time = start_time - shift_sec
            start_bit = int(sr * shifted_time)
        else:
            start_bit = int(sr * start_time)
        end_bit = int(start_bit + sr * self.duration)
        cut_audio = audio[start_bit:end_bit, ]
        assert cut_audio.shape[0] == self.duration * sr, (sr, audio.shape,
                                                          start_time,
                                                          self.duration,
                                                          cut_audio.shape[0])

        return cut_audio

    def find_most_relevent(self, img_path):
        imgs = sorted(os.listdir(img_path[:-9]))
        img_idx = int(img_path[-9:-4])
        imgs_idx = [abs(int(i[-9:-4]) - img_idx) for i in imgs]
        selected = imgs[imgs_idx.index(min(imgs_idx))]
        return img_path[:-9] + selected

    def __len__(self):
        return len(self.video_info)
