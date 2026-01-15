import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        datafiles=[''], 
        sampling_rate=32000, 
        num_channels=1,
        max_clip_len=5,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index):
        try:
            audio_path = self.all_data_json[index]['wav']
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
            text = self.all_data_json[index]['caption']

            # drop short utterance
            if audio_data.size(1) < self.sampling_rate * 0.5:
                raise Exception(f'{audio_path} is too short, drop it ...') 
            
            return text, audio_data, audio_rate
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {audio_path}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)

    def __getitem__(self, index):
        # create a audio tensor  
        text, audio_data, audio_rate = self._read_audio(index)
        audio_len = audio_data.shape[1] / audio_rate
        # convert stero to single channel
        if self.num_channels > 1:
            if audio_data.shape[0] == 1: # fake multi-channel
                audio_data = create_synthetic_multichannel(audio_data, num_channels=4, sr=self.sampling_rate, max_delay_ms=50, gain_range=(0.5, 1.5))
            elif audio_data.shape[0] == self.num_channels:
                pass
            else:
                raise ValueError(f"audio channels {audio_data.shape[0]} not match the required num_channels {self.num_channels}")
            if audio_rate != self.sampling_rate:
                audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
            audio_data = self._cut_or_randomcrop(audio_data)
        else: # mono
            if audio_data.shape[0] > 1:
                # audio_data: [samples]
                audio_data = (audio_data[0] + audio_data[1]) / 2
            else:
                audio_data = audio_data.squeeze(0)
            
            # resample audio clip
            if audio_rate != self.sampling_rate:
                audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
            
            audio_data = audio_data.unsqueeze(0)
            
            audio_data = self._cut_or_randomcrop(audio_data)      
        data_dict = {
            'text': text, 
            'waveform': audio_data,  
            'modality': 'audio_text'
        }

        return data_dict


def create_synthetic_multichannel(audio, num_channels=4, sr=16000, max_delay_ms=50, gain_range=(0.5, 1.5)):
    """
    Create synthetic multi-channel audio with random channel-wise delays and gains.
    
    Args:
        audio: (samples,) or (1, samples) tensor, mono audio input
        num_channels: Number of output channels
        sr: Sample rate
        max_delay_ms: Maximum random delay in milliseconds
        gain_range: Tuple of (min_gain, max_gain) for random scaling
    
    Returns:
        multichannel: (num_channels, samples) tensor
    """
    if audio.dim() > 1:
        audio = audio.squeeze()
    
    max_delay_samples = int(max_delay_ms * sr / 1000)
    output_length = audio.size(0) + max_delay_samples
    multichannel = torch.zeros(num_channels, output_length)
    
    for ch in range(num_channels):
        delay = random.randint(0, max_delay_samples)
        gain = random.uniform(*gain_range)
        multichannel[ch, delay:delay + audio.size(0)] = audio * gain
    
    return multichannel
