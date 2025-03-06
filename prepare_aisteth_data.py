import torchaudio
import torch
import os
import librosa
import pandas as pd
from tqdm import tqdm
import ast


def read_audio_file(filepath, sample_rate):
    y, sr=librosa.load(filepath, sr=sample_rate)
    if sr!=sample_rate:
        y=torchaudio.functional.resample(y, sr, sample_rate)
    return torch.from_numpy(y)

def extract_files_from_intervals(ip_folder, op_folder, audio_file, intervals, sample_rate):
    ip_file_path = os.path.join(ip_folder, audio_file)
    x = read_audio_file(ip_file_path, sample_rate)

    for interval in intervals:
        start, end= interval
        y=x[start*sample_rate:end*sample_rate]
        op_file_name = audio_file.replace(".wav", f"_{interval[0]}_{interval[1]}.wav")
        op_file_path = os.path.join(op_folder, op_file_name)
        y.unsqueeze_(0)
        torchaudio.save(op_file_path, y, sample_rate)


def extract_all_files_from_intervals():
    ip_folder = "unsynced/raw_data/AiSteth_raw"
    op_folder = "unsynced/raw_data/AiSteth"
    # df = pd.read_csv("../data_preparation/k_folds/AiSteth/fold_01.csv")
    df = pd.read_csv("unsynced/data_preparation/k_folds/AiSteth/validation.csv")
    sample_rate = 4000

    for i in tqdm(range(len(df))):
        audio_file = df.iloc[i]["New_Audio_filename"]
        audio_file = f"{audio_file}.wav"
        intervals = ast.literal_eval(df.iloc[i]["chunks_to_crop"])
        extract_files_from_intervals(ip_folder, op_folder, audio_file, intervals, sample_rate)

    print("AiSteth Files extracted successfully")

def get_interval_duration(ip_folder, audio_file, intervals, sample_rate):
    ip_file_path = os.path.join(ip_folder, audio_file)
    x = read_audio_file(ip_file_path, sample_rate)
    duration = 0

    for interval in intervals:
            start, end= interval
            y=x[start*sample_rate:end*sample_rate]
            duration += len(y)/sample_rate
    
    return duration

def calculate_aisteth_data_duration():
    ip_folder = "unsynced/raw_data/AiSteth_raw"
    df = pd.read_csv("unsynced/data_preparation/k_folds/AiSteth/fold_05.csv")
    # df = pd.read_csv("../data_preparation/k_folds/AiSteth/validation.csv")
    sample_rate = 4000 
    normal_files, murmur_files, normal_data_duration, murmur_data_duration = 0, 0, 0, 0
    test_murmur, train_murmur, test_normal, train_normal = 0, 0, 0, 0
    for i in tqdm(range(len(df))):
        set_type = df.iloc[i]["set_type"]
        audio_file = df.iloc[i]["New_Audio_filename"]
        audio_file = f"{audio_file}.wav"
        intervals = ast.literal_eval(df.iloc[i]["chunks_to_crop"])
        duration = get_interval_duration(ip_folder, audio_file, intervals, sample_rate)
        if "Murmur" in audio_file:
            murmur_files += len(intervals)
            murmur_data_duration += duration
            if set_type == "test":
                test_murmur += duration
            else:   
                train_murmur += duration
        else:
            normal_files += len(intervals)
            normal_data_duration += duration
            if set_type == "test":
                test_normal += duration
            else:   
                train_normal += duration

    print(normal_data_duration+murmur_data_duration)
    print(normal_files+murmur_files)

    normal_data_duration = round(normal_data_duration/3600, 2)
    murmur_data_duration = round(murmur_data_duration/3600, 2)
    test_normal = round(test_normal/3600, 2)
    train_normal = round(train_normal/3600, 2)
    test_murmur = round(test_murmur/3600, 2)
    train_murmur = round(train_murmur/3600, 2)
    print(f"Normal files: {normal_files}, Murmur files: {murmur_files}")
    print(f"Normal data duration: {normal_data_duration}, Murmur data duration: {murmur_data_duration}")
    print(f"Train Normal data duration: {train_normal}, Test Normal data duration: {test_normal}")
    print(f"Train Murmur data duration: {train_murmur}, Test Murmur data duration: {test_murmur}")


def get_duration_from_recording_locations(ip_folder, audio_file, recording_locations, sample_rate):
    ip_file_path = os.path.join(ip_folder, audio_file)
    x = read_audio_file(ip_file_path, sample_rate)
    duration = 0

    for recording_location in recording_locations:
        start, end = recording_location
        y = x[start*sample_rate:end*sample_rate]
        duration += len(y)/sample_rate

    return duration

def calculate_physionet_2022_data_duration():
    ip_folder = "unsynced/raw_data/physionet_2022"
    df = pd.read_csv("unsynced/data_preparation/k_folds/physionet_2022/fold_05.csv")
    sample_rate = 4000
    normal_files, murmur_files, normal_data_duration, murmur_data_duration = 0, 0, 0, 0
    test_murmur, train_murmur, test_normal, train_normal = 0, 0, 0, 0
    for i in tqdm(range(len(df))):
        label = df.iloc[i]["Murmur"] 
        if label == "Unknown":
            continue
        audio_file = df.iloc[i]["Patient ID"]
        set_type = df.iloc[i]["set_type"]
        recording_locations = df.iloc[i]["Recording locations:"].split("+")
        if(type(df.iloc[i]["Murmur locations"]) == float):
            murmur_locations = []
        else:
            murmur_locations = df.iloc[i]["Murmur locations"].split("+")

        for location in recording_locations:
            count = recording_locations.count(location)
            if count == 1:
                file = f"{audio_file}_{location}.wav"
                y = read_audio_file(os.path.join(ip_folder, file), sample_rate)
                duration = len(y)/sample_rate
                if location in murmur_locations:
                    murmur_files += 1
                    murmur_data_duration += duration
                    if set_type == "test":
                        test_murmur += duration
                    else:   
                        train_murmur += duration
                else:
                    normal_files += 1
                    normal_data_duration += duration
                    if set_type == "test":
                        test_normal += duration
                    else:
                        train_normal += duration
            else:
                for i in range(count):
                    file = f"{audio_file}_{location}_{i+1}.wav"
                    y = read_audio_file(os.path.join(ip_folder, file), sample_rate)
                    duration = len(y)/sample_rate
                    if location in murmur_locations:
                        murmur_files += 1
                        murmur_data_duration += duration
                        if set_type == "test":
                            test_murmur += duration
                        else:   
                            train_murmur += duration
                    else:
                        normal_files += 1
                        normal_data_duration += duration
                        if set_type == "test":
                            test_normal += duration
                        else:
                            train_normal += duration


    print(normal_data_duration+murmur_data_duration)
    print(normal_files+murmur_files)


    normal_data_duration = round(normal_data_duration/3600, 2)
    murmur_data_duration = round(murmur_data_duration/3600, 2)
    test_normal = round(test_normal/3600, 2)
    train_normal = round(train_normal/3600, 2)
    test_murmur = round(test_murmur/3600, 2)
    train_murmur = round(train_murmur/3600, 2)
    print(f"Normal files: {normal_files}, Murmur files: {murmur_files}")
    print(f"Normal data duration: {normal_data_duration}, Murmur data duration: {murmur_data_duration}")
    print(f"Test Normal data duration: {test_normal}, Train Normal data duration: {train_normal}")
    print(f"Test Murmur data duration: {test_murmur}, Train Murmur data duration: {train_murmur}")


if __name__ == "__main__":
    calculate_aisteth_data_duration()
    calculate_physionet_2022_data_duration()
