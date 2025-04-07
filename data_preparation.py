from ast import literal_eval
import soundfile as sf
import os, glob, argparse, torch, torchaudio, random, logging, librosa
from scipy.signal import butter, lfilter
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils.wav2vec2_inference import extract_features
import ast
import matplotlib.pyplot as plt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa

def save_wav2vec2_features(df, device, model, feature_dir, feature_type):
    for _, row in tqdm(df.iterrows()):
            filepath=row['filepath']
            label=row['label']
            set_type, filename=filepath.rsplit('/',2)[-2:]
            y, sr=torchaudio.load(filepath)
            y=y.to(device)
            features=extract_features(model=model, input_tensor=y, feature_type=feature_type)
            torch.save(features, os.path.join(feature_dir, set_type, str(label), filename[:-4]+".pt"))

def extract_wav2vec2_features(model, input_dir, output_dir, feature_type, device, validation=False):
    if validation:
        dataset = "AiSteth"
        data_directory = os.path.join(input_dir, dataset)
        feature_dir = os.path.join(output_dir, dataset)
        df=pd.read_csv(os.path.join(data_directory, "validation-files.csv"))
        for label in df['label'].unique():
            os.makedirs(os.path.join(feature_dir, 'validation', str(label)), exist_ok=False)
        save_wav2vec2_features(df, device, model, feature_dir, feature_type)
    else:
        datasets = ["physionet_2022", "AiSteth"]
        for dataset in datasets:
            data_directory = os.path.join(input_dir, dataset)
            feature_dir = os.path.join(output_dir, dataset)
            df=pd.read_csv(os.path.join(data_directory, "train-test-files.csv"))
            for label in df['label'].unique():
                os.makedirs(os.path.join(feature_dir, 'train', str(label)), exist_ok=False)
                os.makedirs(os.path.join(feature_dir, 'test', str(label)), exist_ok=False)

            save_wav2vec2_features(df, device, model, feature_dir, feature_type)

def extract_w_features(model, input_tensor, feature_type):
    with torch.no_grad():
        # outputs=model(input_tensor)
        # outputs = model.extract_features(input_tensor)
        # encoder = model.model.encoder
        # outputs = encoder(input_tensor)
        # if feature_type=="transformer":
        #     return outputs.last_hidden_state
        # elif feature_type=="cnn":
        #     return outputs.extract_features
        # else:
        #     raise ValueError("Invalid feature type. Choose either 'transformer' or 'cnn'.")
        print(input_tensor.shape)
        input_tensor = input_tensor.unsqueeze(0)
        print(input_tensor.shape)
        outputs = model.encoder(input_tensor)
        features = outputs.last_hidden_state
        print(features.shape)
        return features

def save_whisper_features(df, device, feature_dir):
    torch_dtype = torch.float32
    model_id = "distil-whisper/distil-small.en"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)
    encoder = model.get_encoder()
    processor = AutoProcessor.from_pretrained(model_id)
    for _, row in tqdm(df.iterrows()):
            filepath=row['filepath']
            label=row['label']
            set_type, filename=filepath.rsplit('/',2)[-2:]
            sample, sr=librosa.load(filepath, sr=16000)
            input_features = processor(sample, sampling_rate = 16000, return_tensors="pt").input_features
            input_features = input_features.to(device, dtype=torch_dtype)
            with torch.no_grad():
                features = encoder(input_features).last_hidden_state
            torch.save(features, os.path.join(feature_dir, set_type, str(label), filename[:-4]+".pt"))

def extract_whisper_features(input_dir, output_dir, device, validation=False):
    if validation:
        dataset = "AiSteth"
        data_directory = os.path.join(input_dir, dataset)
        feature_dir = os.path.join(output_dir, dataset)
        df=pd.read_csv(os.path.join(data_directory, "validation-files.csv"))
        for label in df['label'].unique():
            os.makedirs(os.path.join(feature_dir, 'validation', str(label)), exist_ok=False)
        save_whisper_features(df, device, feature_dir)
    else:
        datasets = ["physionet_2022", "AiSteth"]
        # datasets = ["physionet_2022"]
        for dataset in datasets:
            data_directory = os.path.join(input_dir, dataset)
            feature_dir = os.path.join(output_dir, dataset)
            df=pd.read_csv(os.path.join(data_directory, "train-test-files.csv"))
            for label in df['label'].unique():
                os.makedirs(os.path.join(feature_dir, 'train', str(label)), exist_ok=False)
                os.makedirs(os.path.join(feature_dir, 'test', str(label)), exist_ok=False)

            save_whisper_features(df, device, feature_dir)


def save_hubert_features(df, device, feature_dir):
    torch_dtype = torch.float32
    model_id = "distil-whisper/distil-small.en"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)
    encoder = model.get_encoder()
    processor = AutoProcessor.from_pretrained(model_id)
    for _, row in tqdm(df.iterrows()):
            filepath=row['filepath']
            label=row['label']
            set_type, filename=filepath.rsplit('/',2)[-2:]
            sample, sr=librosa.load(filepath, sr=16000)
            input_features = processor(sample, sampling_rate = 16000, return_tensors="pt").input_features
            input_features = input_features.to(device, dtype=torch_dtype)
            with torch.no_grad():
                features = encoder(input_features).last_hidden_state
            torch.save(features, os.path.join(feature_dir, set_type, str(label), filename[:-4]+".pt"))

def extract_hubert_features(input_dir, output_dir, device, validation=False):
    if validation:
        dataset = "AiSteth"
        data_directory = os.path.join(input_dir, dataset)
        feature_dir = os.path.join(output_dir, dataset)
        df=pd.read_csv(os.path.join(data_directory, "validation-files.csv"))
        for label in df['label'].unique():
            os.makedirs(os.path.join(feature_dir, 'validation', str(label)), exist_ok=False)
        save_hubert_features(df, device, feature_dir)
    else:
        datasets = ["physionet_2022", "AiSteth"]
        # datasets = ["physionet_2022"]
        for dataset in datasets:
            data_directory = os.path.join(input_dir, dataset)
            feature_dir = os.path.join(output_dir, dataset)
            df=pd.read_csv(os.path.join(data_directory, "train-test-files.csv"))
            for label in df['label'].unique():
                os.makedirs(os.path.join(feature_dir, 'train', str(label)), exist_ok=False)
                os.makedirs(os.path.join(feature_dir, 'test', str(label)), exist_ok=False)

            save_hubert_features(df, device, feature_dir)

def extract_mel_spectrogram(y, sample_rate, n_mels=200, hop_length=64, n_fft=256):
    mel_spectrogram=librosa.feature.melspectrogram(y=y.numpy(), sr=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spectrogram=torch.tensor(mel_spectrogram, dtype=torch.float32)
    return mel_spectrogram

def save_mel_spectrogram(df, feature_dir, sample_rate):
    for _, row in tqdm(df.iterrows()):
        filepath=row['filepath']
        label=row['label']
        set_type, filename=filepath.rsplit('/',2)[-2:]
        y, sr=torchaudio.load(filepath)
        mel_spectrogram=extract_mel_spectrogram(y, sample_rate)
        torch.save(mel_spectrogram, os.path.join(feature_dir, set_type, str(label), filename[:-4]+".pt"))

def extract_mel_spectrogram_features(input_dir, output_dir, sample_rate, validation=False):
    if validation:
        dataset = "AiSteth"
        data_directory = os.path.join(input_dir, dataset)
        feature_dir = os.path.join(output_dir, dataset)
        df=pd.read_csv(os.path.join(data_directory, "validation-files.csv"))
        for label in df['label'].unique():
            os.makedirs(os.path.join(feature_dir, 'validation', str(label)), exist_ok=False)
        save_mel_spectrogram(df, feature_dir, sample_rate)
    else:
        datasets = ["physionet_2022", "AiSteth"]
        # datasets = ["physionet_2022"]
        for dataset in datasets:
            data_directory = os.path.join(input_dir, dataset)
            feature_dir = os.path.join(output_dir, dataset)
            df=pd.read_csv(os.path.join(data_directory, "train-test-files.csv"))
            for label in df['label'].unique():
                os.makedirs(os.path.join(feature_dir, 'train', str(label)), exist_ok=False)
                os.makedirs(os.path.join(feature_dir, 'test', str(label)), exist_ok=False)

            save_mel_spectrogram(df, feature_dir, sample_rate)

def extract_mfcc(y, sample_rate, n_mfcc=25, hop_length=64, n_fft=256):
    mfcc=librosa.feature.mfcc(y=y.numpy(), sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfcc=torch.tensor(mfcc, dtype=torch.float32)
    return mfcc

def save_mfcc(df, feature_dir, sample_rate):
    for _, row in tqdm(df.iterrows()):
        filepath=row['filepath']
        label=row['label']
        set_type, filename=filepath.rsplit('/',2)[-2:]
        y, sr=torchaudio.load(filepath)
        mfcc=extract_mfcc(y, sample_rate)
        torch.save(mfcc, os.path.join(feature_dir, set_type, str(label), filename[:-4]+".pt"))

def extract_mfcc_features(input_dir, output_dir, sample_rate, validation=False):
    if validation:
        dataset = "AiSteth"
        data_directory = os.path.joinnoise(input_dir, dataset)
        feature_dir = os.path.join(output_dir, dataset)
        df=pd.read_csv(os.path.join(data_directory, "validation-files.csv"))
        for label in df['label'].unique():
            os.makedirs(os.path.join(feature_dir, 'validation', str(label)), exist_ok=False)
        save_mfcc(df, feature_dir, sample_rate)
    else:
        datasets = ["physionet_2022", "AiSteth"]
        # datasets = ["physionet_2022"]
        for dataset in datasets:
            data_directory = os.path.join(input_dir, dataset)
            feature_dir = os.path.join(output_dir, dataset)
            df=pd.read_csv(os.path.join(data_directory, "train-test-files.csv"))
            for label in df['label'].unique():
                os.makedirs(os.path.join(feature_dir, 'train', str(label)), exist_ok=False)
                os.makedirs(os.path.join(feature_dir, 'test', str(label)), exist_ok=False)

            save_mfcc(df, feature_dir, sample_rate)

def normalize(signal):
    return (signal - signal.mean()) / signal.std()

def read_audio_file(filepath, sample_rate):
    # print(filepath)
    # y, sr=librosa.load(filepath, sr=None)
    # print(sr, sample_rate)
    y, sr=librosa.load(filepath, sr=sample_rate)
    # print(sr, sample_rate)
    if sr!=sample_rate:
        y=torchaudio.functional.resample(y, sr, sample_rate)
    return torch.from_numpy(y)

def original(signal, sample_rate, chunk_duration):
    signal_duration=(signal.numel()/sample_rate)
    if signal.ndim!=1:
        raise ValueError(f"Expected 1 dimensional tensor.")
    
    signals=[]
    step=sample_rate*chunk_duration
    if signal_duration<3:
        return signals

    if signal_duration<chunk_duration:
        tmp=signal
        while len(tmp)<step:
            tmp=torch.concatenate([tmp, signal], dim=0)
        signals.append([tmp[:step], -1])
        return signals
    
    n_chunks, rem=divmod(signal_duration, chunk_duration)
    for i in range(int(n_chunks)):
        signals.append([signal[i*sample_rate*chunk_duration:step+(i*sample_rate*chunk_duration), i*chunk_duration]])
    if n_chunks and rem:
        signals.append([signal[-step:], signal_duration-chunk_duration])
    return signals

def with_hop(signal, sample_rate, chunk_duration, hop_duration=None):
    signal_duration=(signal.shape[-1]/sample_rate)
    if signal.ndim!=1:
        raise ValueError(f"Expected 1 dimensional tensor with signal duration({signal_duration})>=chunk_duration({chunk_duration}).")
    signals=[]
    step=sample_rate*chunk_duration

    if signal_duration>=chunk_duration:
        for start in np.arange(0, int(signal_duration)-chunk_duration+0.00001, hop_duration):
            start=int(start)
            signals.append([signal[start*sample_rate:step+(start*sample_rate)], round(start,3)])
    elif signal_duration>=3 and signal_duration<chunk_duration:
        tmp=signal
        while len(tmp)<step:
            tmp=torch.concatenate([tmp, signal], dim=0)
        signals.append([tmp[:step], -1])
    return signals

def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_butter_bandpass(signal, sample_rate, lowcut=20, highcut=450, order=4):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = lfilter(b, a, signal)
    return y

def add_white_noise(sig, target_snr_db, mean_noise = 0 ):
    # Calculate signal power and convert to dB 
    sig_amplitude = np.mean(sig**2)
    sig_avg_db = 10 * np.log10(sig_amplitude)
    # Calculate noise according to targeted SNR
    noise_avg_db = sig_avg_db - target_snr_db
    noise_amplitude = 10 ** (noise_avg_db / 20)
    # Generate an sample of white noise
    noisy_sig = sig+np.random.normal(mean_noise, noise_amplitude, len(sig))
    return noisy_sig

def preprocess(signal, sample_rate):
    signal=signal.numpy()
    signal=add_white_noise(signal, target_snr_db=10, mean_noise=0)
    signal=normalize(signal)
    signal=apply_butter_bandpass(signal, sample_rate, lowcut=20, highcut=450, order=4)
    return torch.tensor(signal, dtype=torch.float32)

def get_label(filepath):
    test_class_labels={"test_noise_data":0, "test_normal_data":1, "test_abnormal_data":2}
    _, class_label, filename=filepath.rsplit('/',2)
    return test_class_labels[class_label]

def main(all_filepaths, chunk_duration, sample_rate, save_directory, datatype, logger, csv_filepath, hop=None, validation=False):
    # Validations
    logger.debug(f"chunk_duration: {chunk_duration}, sample_rate:{sample_rate}, save_directory:{save_directory}, datatype:{datatype}, hop:{hop}")
    if datatype=='original':
        pass
    elif datatype=='with_hop':
        assert hop>0, "Hop expected to be positive integer. Got {}".format(hop)
    else:
        raise ValueError(f"Got unexpected value for datatype. One of (original, with_hop).")

    if all_filepaths:
        logger.info(f"Total files found: {len(all_filepaths)}")
    else:
        logger.critical(f"Total files found: {len(all_filepaths)}")

    # Read the audio files.
    n_chunk=0
    data=[]
    if not validation:
        os.makedirs(os.path.join(save_directory, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_directory, 'test'), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_directory, 'validation'), exist_ok=True)
    
    for filepath, label, section in tqdm(all_filepaths):
        try:
            y=read_audio_file(filepath=filepath, sample_rate=sample_rate)
        except:
            import traceback
            traceback.print_exc()
            break
            print(f'Skipping `{filepath}`')
            continue

        if datatype=='original':
            ys=original(signal=y, sample_rate=sample_rate, chunk_duration=chunk_duration)
        elif datatype=='with_hop':
            ys=with_hop(signal=y, sample_rate=sample_rate, chunk_duration=chunk_duration, hop_duration=hop)
        else:
            ys=[]

        for y, start in ys:
            filename=filepath.rsplit('/',1)[-1]
            y=preprocess(y, sample_rate)
            n_chunk+=1
            chunk_name=f"chunk_{str(n_chunk).zfill(6)}.wav"
            chunk_path=os.path.join(save_directory, section, chunk_name)
            y_duration=y.numel()/sample_rate
            if y_duration==chunk_duration:
                y.unsqueeze_(0)
                # print(chunk_path)
                torchaudio.save(chunk_path, y, sample_rate=sample_rate)
                tmp=[chunk_path, label, section, start, filename]
                data.append(tmp)
                if (n_chunk%10000)==0:
                    logger.info(f"{n_chunk} chunks created.")
            else:
                logger.critical(f"{filepath} : {y_duration} seconds. Expected {chunk_duration} seconds.")
    if n_chunk:
        logger.info(f"Successfully created {n_chunk} chunks.")
        df=pd.DataFrame(data, columns=['filepath', 'label', 'set_type', 'start', 'filename'])
        df.to_csv(csv_filepath, index=False)
        logger.info(f"csv {df.shape} saved to `{csv_filepath}`.")
    else:
        logger.critical(f"Created {n_chunk} chunks.")
        df=pd.DataFrame()
    return df


def generate_chunks_aisteth(train_test_csv_filepath, fold_directory, chunk_duration, sample_rate, hop, read_prefix="unsynced/raw_data/AiSteth", validation = False):
    df=pd.read_csv(f"{train_test_csv_filepath}.csv")
    all_filepaths=[]
    labels={'Murmur':0, "Normal":1}

    datatype="with_hop"
    tmp_save_directory=os.path.join(fold_directory, "AiSteth")

    for index, values in df.iterrows():
        intervals  = ast.literal_eval((values['chunks_to_crop']))
        for start, end in intervals:
            filename = f"{values['New_Audio_filename']}_{start}_{end}.wav"
            set_type=values['set_type']
            label=labels[values['label']]
            all_filepaths.append([os.path.join(read_prefix, filename), label, set_type])

    print("Number of files:", len(all_filepaths))
    
    # -----------------------------------------------------
    os.makedirs(tmp_save_directory, exist_ok=True)
    log_filepath=os.path.join(tmp_save_directory, "logs.log")
    if validation:
        csv_filepath=os.path.join(tmp_save_directory, "validation-files.csv")
    else:
        csv_filepath=os.path.join(tmp_save_directory, "train-test-files.csv")

    logging.basicConfig(filename=log_filepath, format='%(asctime)s : %(message)s', filemode='w')
    print("Find logs -->", log_filepath)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    df=main(all_filepaths=all_filepaths, chunk_duration=chunk_duration, sample_rate=sample_rate, datatype=datatype, hop=hop, logger=logger, save_directory=tmp_save_directory, csv_filepath=csv_filepath, validation=validation)


def generate_chunks_physionet(train_test_csv_filepath, fold_directory, chunk_duration, sample_rate, hop, read_prefix="unsynced/raw_data/physionet_2022"):
    df=pd.read_csv(f"{train_test_csv_filepath}.csv")
    all_filepaths=[]
    labels={'Present':0, "Absent":1, "Unknown":2}

    datatype="with_hop"
    tmp_save_directory=os.path.join(fold_directory,"physionet_2022")
        
    for index, values in df.iterrows():
        filename, location, murmu_locs = str(values['Patient ID']), values['Recording locations:'], values['Murmur locations']
        tlabel=values['Murmur']
        set_type=values['set_type']
        ls=location.split('+')
        if isinstance(murmu_locs, str):
            murmur_ls=murmu_locs.split('+')
        else:
            murmur_ls=[]

        for location in ls:
            if location in murmur_ls:
                label="Present"
            elif tlabel=="Unknown":
                label='Unknown'
            else:
                label="Absent"
            label=labels[label]
            if label==2: # Skip unknown labels
                continue
            count=ls.count(location)
            if count==1:
                l=location
                location=f"{filename}_{l}.wav"
                f=location
                f=os.path.join(read_prefix, f)
                all_filepaths.append([f, label, set_type])
            else:
                for i in range(count):
                    l=f"{location}_{i+1}"
                    tmp_location=f"{filename}_{l}.wav"
                    f=tmp_location
                    f=os.path.join(read_prefix, f)
                    all_filepaths.append([f, label, set_type])

    print("Number of files:", len(all_filepaths))

    # -----------------------------------------------------
    os.makedirs(tmp_save_directory, exist_ok=True)
    log_filepath=os.path.join(tmp_save_directory, "logs.log")
    csv_filepath=os.path.join(tmp_save_directory, "train-test-files.csv")

    logging.basicConfig(filename=log_filepath, format='%(asctime)s : %(message)s', filemode='w')
    print("Find logs -->", log_filepath)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    df=main(all_filepaths=all_filepaths, chunk_duration=chunk_duration, sample_rate=sample_rate, datatype=datatype, hop=hop, logger=logger, save_directory=tmp_save_directory, csv_filepath=csv_filepath)


def create_train_test_data(chunk_duration, sample_rate, hop, train_test_csv_folderpath, save_directory, fold):
    fold_directory = os.path.join(save_directory, fold)
    datasets = ["physionet_2022", "AiSteth"]
    # datasets = ["physionet_2022"]

    if(os.path.exists(fold_directory)):
        print(f"Data already exists in  {fold_directory}")
        return
    
    for dataset in datasets:
        train_test_csv_filepath = os.path.join(train_test_csv_folderpath, dataset, fold)
        if dataset=="physionet_2022":   
            generate_chunks_physionet(train_test_csv_filepath, fold_directory, chunk_duration, sample_rate, hop, read_prefix="unsynced/raw_data/physionet_2022")
        elif dataset=="AiSteth":
            generate_chunks_aisteth(train_test_csv_filepath, fold_directory, chunk_duration, sample_rate, hop, read_prefix="unsynced/raw_data/AiSteth")
        else:
            raise ValueError(f"Got unexpected value for dataset. One of (physionet_2022, AiSteth).")
        
def create_validation_data(chunk_duration, sample_rate, hop, train_test_csv_folderpath, save_directory, fold):
    fold_directory = os.path.join(save_directory, fold)

    if(os.path.exists(fold_directory)):
        print(f"Data already exists in  {fold_directory}")
        return
    
    validation_csv_filepath = os.path.join(train_test_csv_folderpath, "AiSteth", fold)
    generate_chunks_aisteth(validation_csv_filepath, fold_directory, chunk_duration, sample_rate, hop, read_prefix="unsynced/raw_data/AiSteth", validation = True)

        
        
