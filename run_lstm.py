import torch,json
import os
from data_preparation import create_train_test_data, extract_wav2vec2_features, create_validation_data, extract_mel_spectrogram_features, extract_mfcc_features, extract_whisper_features
from train_test_evaluate import perform_downstream_task
from utils.wav2vec2_inference import load_model
from transformers import Wav2Vec2Model, WhisperFeatureExtractor, WhisperForConditionalGeneration


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
    print(device)
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
    base_model = load_model(model_path, config_path, device)
    # base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    # model_path = "unsynced/models/ssamba_base_250.pth"
    model_path = "unsynced/models/base_scratch-audioset-32.74.pth"
    #load model
    # base_model = torch.load(model_path, weights_only=False)
    # base_model = Wav2Vec2BertModel.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
    # base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    #remove last layer
    # base_model = torch.nn.Sequential(*list(base_model.children())[:-1])



    # base_model = base_model.to(device)
    # print(base_model)

    # Extract the features.
    input_dir = os.path.join(save_directory, fold)
    # output_dir = os.path.join(save_directory, "base_model_1", fold)
    feature_extraction_technique = "wav2vec2-base"
    # feature_extraction_technique = "whisper_small"
    output_dir = os.path.join(save_directory, feature_extraction_technique, fold)

    print()
    print(f"Extracting features from {input_dir} to {output_dir}")
    if(os.path.exists(output_dir)):
        print(f"Features already extracted in {output_dir}, Delete the folder to re-extract the features")
    else:
        if feature_extraction_technique == "wav2vec" or feature_extraction_technique == "wav2vec2-base" or feature_extraction_technique == "wav2vec2-base_corrected_sr":
            extract_wav2vec2_features(model=base_model, input_dir = input_dir, output_dir= output_dir, feature_type="transformer", device=device)
            
        # extract_wav2vec2_features(model=base_model, input_dir = input_dir, output_dir= output_dir, feature_type="transformer", device=device)  
        extract_whisper_features(input_dir = input_dir, output_dir= output_dir, device=device) 
        # extract_mel_spectrogram_features(input_dir, output_dir, sample_rate)   
        # extract_mfcc_features(input_dir, output_dir, sample_rate)                      


    # Perform the downstream task.
    print()
    print("Model training and evaluation started.")
    perform_downstream_task(model_training_config, device, fold)

    # # Validation data preparation
    # fold = "validation"

    # create_validation_data(chunk_duration, sample_rate, hop, train_test_csv_folderpath, save_directory, fold)

    # input_dir = os.path.join(save_directory, fold)
    # output_dir = os.path.join(save_directory, "base_model_1", fold)

    # print(input_dir, output_dir)
    # extractwav2vec2_features(model=base_model, input_dir = input_dir, output_dir= output_dir, feature_type="transformer", device=device, validation=True)
