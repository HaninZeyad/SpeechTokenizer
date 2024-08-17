from transformers import AutoProcessor, TFHubertModel
from datasets import load_dataset
import soundfile as sf
from transformers import AutoProcessor, HubertModel
import torch
from transformers import Wav2Vec2FeatureExtractor 
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
# model = HubertModel.from_pretrained("facebook/hubert-base-ls960",from_tf=True)

# processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960", cache_dir="/ocean/projects/cis220031p/hatwany/cache")
# model = HubertModel.from_pretrained("facebook/hubert-base-ls960",from_tf=True, cache_dir="/ocean/projects/cis220031p/hatwany/cache")
# processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
#model = HubertModel.from_pretrained("facebook/hubert-base-ls960",from_tf=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960", cache_dir="/ocean/projects/cis220031p/hatwany/cache")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", cache_dir="/ocean/projects/cis220031p/hatwany/cache")
# Print out some configuration details
print("Model Configuration:")
print(f"Hidden Size: {model.config.hidden_size}")
print(f"Num Attention Heads: {model.config.num_attention_heads}")
print(f"Num Hidden Layers: {model.config.num_hidden_layers}")
print(f"Intermediate Size: {model.config.intermediate_size}")

# Define the path to save the model and configuration
save_path = '/ocean/projects/cis220031p/hatwany/SpeechTokenizer/hubert_base'

# Save the model and configuration locally
model.save_pretrained(save_path)
model.config.save_pretrained(save_path)
feature_extractor.save_pretrained(save_path)
