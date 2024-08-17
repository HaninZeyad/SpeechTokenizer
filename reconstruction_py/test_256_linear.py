import sys
import torch
import torch.nn.functional as F
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
                 config_path='/ocean/projects/cis220031p/hatwany/SpeechTokenizer/Log/spt_256_10_linear/config.json',
                 ckpt_path='/ocean/projects/cis220031p/hatwany/SpeechTokenizer/Log/spt_256_10_linear/SpeechTokenizer_best_dev.pt'):
        super().__init__()
        self.out_prefix = config_path.split("/")[-2]
        self.model = _SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.sample_rate = self.model.sample_rate
        self.pretrained = pretrained
        self.frozen = frozen
        if self.frozen:
            self.model.eval()
    
    def encode(self, x):
        # Adjust input shape
        expected_dim = 1024  # Adjust this value based on your model's expected input dimension
        current_dim = x.shape[-1]
        if current_dim < expected_dim:
            x = F.pad(x, (0, 0, 0, expected_dim - current_dim))
        elif current_dim > expected_dim:
            x = x[..., :expected_dim]

        # Transpose the tensor if needed (assuming the model expects [batch, features, time])
        x = x.transpose(1, 2)  # [batch, time, features] -> [batch, features, time]

        if self.frozen:
            with torch.no_grad():
                token = self.model.encode(x)
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

def tokenize_speech(wav_path, tokenizer):
    wav, sr = torchaudio.load(wav_path)
    if sr != tokenizer.sample_rate:
        wav = resample(wav, sr, tokenizer.sample_rate)
    wav = wav.unsqueeze(0)  # Add batch dimension
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
        audio_data = y_pred.squeeze().cpu().numpy()
        
        # Normalize the audio data to the range [-1, 1]
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Debugging: Print shapes
        print(f"Shape of y_pred: {y_pred.shape}")
        print(f"Shape of audio_data: {audio_data.shape}")

        # Play the audio
        display(Audio(audio_data, rate=tokenizer.sample_rate))
        
        # Save the audio file
        sf.write(f'{out_prefix}_{end_codebook_index}.wav', audio_data, tokenizer.sample_rate)

def main():
    tokenizer = SpeechTokenizer_Class()
    audio_path = "/ocean/projects/cis220031p/hatwany/SpeechTokenizer/samples/YOU0000000299_S0000128.wav"
    
    try:
        process_and_save_audio(audio_path, tokenizer)
        print("Audio processing and saving completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Additional error information
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
