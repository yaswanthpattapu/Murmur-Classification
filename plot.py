import json, torch, os
import torchaudio
from utils.wav2vec2_inference import load_model, extract_features
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


with open("utils/downstream_task_config.json", "r") as f:
        config=json.load(f)

model_path = config["base_model"]["model_path"]
config_path = config["base_model"]["config_path"]
device = config["downstream_task"]["model_training"]["training_params"]["device"]
if device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(model_path, config_path, device)

filepath=os.path.join("unsynced", "data", "fold_01", "train", "chunk_000001.wav") #Chunk with murmurs
# filepath = "unsynced/data/fold_01/train/chunk_000090.wav" #Chunk without any murmurs

# label=row['label']
set_type, filename=filepath.rsplit('/',2)[-2:]
y, sr=torchaudio.load(filepath)
y=y.to(device)
features=extract_features(model=model, input_tensor=y, feature_type="transformer")
array = features[0].cpu().numpy().T
plt.imshow(array, aspect='auto', origin='lower', cmap='magma')
plt.savefig("unsynced/pics/chunk_000001.png")
# print(weights)





