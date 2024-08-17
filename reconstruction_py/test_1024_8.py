import sys 
import torch
from speechtokenizer import SpeechTokenizer as _SpeechTokenizer
import torchaudio
from torchaudio.functional import resample
import numpy as np
from IPython.display import display, Audio
import soundfile as sf
from tqdm import tqdm

class SpeechTokenizer_Class():
    def __init__(self, pretrained: bool=True, 
                 frozen: bool=True,
                 config_path='/ocean/projects/cis220031p/hatwany/SpeechTokenizer/Log/spt_1024/config.json',
                 ckpt_path='/ocean/projects/cis220031p/hatwany/SpeechTokenizer/Log/spt_1024/SpeechTokenizer_best_dev.pt'):
        super().__init__()
        self.out_prefix = config_path.split("/")[-2]
        self.model = _SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.sample_rate = self.model.sample_rate
        self.pretrained = pretrained
        self.frozen = frozen
        if self.frozen:
            self.model.eval()
    
    def encode(self, x):
        if self.frozen:
            with torch.no_grad():
                token = self.model.encode(x) # (n_q, B, T)
        else: 
            token = self.model.encode(x)
        return token

    def decode(self, x):
        if self.frozen:
            with torch.no_grad():
                wav = self.model.decode(x)
        else: 
            wav = self.model.decode(x)
        return wav

tokenizer = SpeechTokenizer_Class()

def tokenize_speech(wav_path, tokenizer):
    wav, sr = torchaudio.load(wav_path)
    if sr != tokenizer.sample_rate:
        wav = resample(wav, sr, tokenizer.sample_rate)
    wav = wav.unsqueeze(0)
    codes = tokenizer.encode(wav)
    return codes, wav

def process_and_save_audio(audio_path, tokenizer):
    codes, y_true = tokenize_speech(audio_path, tokenizer)
    out_prefix = tokenizer.out_prefix

    start_codebook_index = 0

    for i in tqdm(range(1, 9)):
        end_codebook_index = i
        
        y_pred = tokenizer.decode(codes[start_codebook_index:end_codebook_index])

        # Extract the audio data from the tensor
        audio_data = y_pred.squeeze().numpy()
        
        # Normalize the audio data to the range [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Play the audio
        display(Audio(audio_data, rate=16000))
        
        # Save the audio file
        sf.write(f'{out_prefix}_{end_codebook_index}.wav', audio_data, 16000)

# Usage
audio_path = "/ocean/projects/cis220031p/hatwany/SpeechTokenizer/samples/YOU0000000299_S0000128.wav"
process_and_save_audio(audio_path, tokenizer)