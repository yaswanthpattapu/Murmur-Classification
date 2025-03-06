import torch, json
from transformers import Wav2Vec2Config, Wav2Vec2Model

def load_model(model_path, config_path, device):
    with open(config_path, 'r') as f:
        config=json.load(f)
    config=Wav2Vec2Config().from_dict(config)
    model=Wav2Vec2Model(config=config).from_pretrained(pretrained_model_name_or_path=model_path, config=config)
    model=model.to(device)
    return model.eval()

def extract_features(model, input_tensor, feature_type):
    with torch.no_grad():
        outputs=model(input_tensor)
        if feature_type=="transformer":
            return outputs.last_hidden_state
        elif feature_type=="cnn":
            return outputs.extract_features
        else:
            raise ValueError("Invalid feature type. Choose either 'transformer' or 'cnn'.")

if __name__=="__main__":
    model_path="models/____/____.pt"
    config_path="models/____/____.json"
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_model(model_path=model_path, config_path=config_path, device=device)

    features=extract_features(model=model, input_tensor=torch.randn(2, 240000).to(device), feature_type="transformer")
