import torch,json
import os
from data_preparation import create_train_test_data
from train_test_evaluate import perform_downstream_task, perform_downstream_task_with_model
from utils.wav2vec2_inference import load_model
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

if __name__ == "__main__":
    
    with open("utils/downstream_task_config.json", "r") as f:
        config=json.load(f)

    # Data preparation parameters
    data_preparation_config = config["downstream_task"]["data_preparation"]
    train_test_csv_folderpath = data_preparation_config["train_test_csv_folderpath"]
    fold = data_preparation_config["k_fold"]
    chunk_duration = data_preparation_config["chunk_duration"]
    hop = data_preparation_config["hop"]
    sample_rate = data_preparation_config["sample_rate"]
    save_directory = data_preparation_config["save_directory"] 

    #Feature extraction parameters
    model_path = config["base_model"]["model_path"]
    config_path = config["base_model"]["config_path"]
    device = config["downstream_task"]["model_training"]["training_params"]["device"]
    if (device == "cuda") or (device.startswith("cuda:") and device[5:].isdigit()):
        device = device if torch.cuda.is_available() else "cpu"

    # Model training parameters
    model_training_config = config["downstream_task"]["model_training"]

    # Create the train and test data.
    print()
    print(f"Creating train and test data for {fold}")

    create_train_test_data(chunk_duration, sample_rate, hop, train_test_csv_folderpath, save_directory, fold)

    # Load the base model.

    print()
    print(f"Loading the base model from {model_path}")
    # base_model = load_model(model_path, config_path, device)


    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)


    # print(base_model)

    #print output shape of the model
    # print(base_model(torch.randn(1, 10000)).last_hidden_state.shape)
    # print(base_model.final_layer_norm)


    # Perform the downstream task.
    print()
    print("Model training and evaluation started.")
    perform_downstream_task_with_model(base_model, model_training_config, device, fold)

