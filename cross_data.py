import torch
from torch.utils.data import DataLoader
from train_test_evaluate import MyDataset, calc_wt_accuracy
from sklearn.metrics import confusion_matrix
from validation import get_labels
import pandas as pd
import matplotlib.pyplot as plt

from utils.lstm_scaled_attn import LSTMWithScaledDotProductAttention
from train_test_evaluate import load_downstream_model

batch_size = 32
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def chunk_level_validation_accuracy(model, test_dataloader):
    val_actuals ,val_predictions = get_labels(model, test_dataloader)
    count = 0
    # count = count + val_actuals == val_predictions
    for i in range(len(val_actuals)):
        if val_actuals[i] == val_predictions[i]:
            count += 1
        

    cf_matrix=confusion_matrix(val_actuals, val_predictions)
    weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[5, 1])

    return weighted_accuracy

def file_level_accuracy(model, test_dataloader, fold, test_dataset):
    val_actuals ,val_predictions = get_labels(model, test_dataloader)
    csv_file = f"unsynced/data/{fold}/{test_dataset}/train-test-files.csv"
    df = pd.read_csv(csv_file)
    df = df[df["set_type"]=="test"]
    chunk_names = df["filepath"].tolist()
    file_names = df["filename"].tolist()
    unique_files = list(set(file_names))
    file_actuals = [df[df["filename"]==file]["label"].tolist()[0] for file in unique_files]
    file_chunks = []


    for i in range(len(chunk_names)):
        chunk_names[i] = chunk_names[i].split("/")[-1]

    file_predictions = []
    for file in unique_files:
        pred = 0
        indices = [i for i, x in enumerate(file_names) if x == file]
        labels = [val_predictions[i] for i in indices]
        actuals = [val_actuals[i] for i in indices]
        murmur_count = labels.count(0)
        # print(labels, actuals)
        cnt = 0
        for i in range(len(labels)):
            if labels[i] == actuals[i]:
                cnt += 1
        file_chunks.append(cnt)
        if murmur_count/len(labels) >= 0.5:
            file_predictions.append(0)
        else:
            file_predictions.append(1)

    # cf_matrix=confusion_matrix(val_actuals, val_predictions)
    # weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[5, 1])
    # print(weighted_accuracy)

    # print(file_actuals, file_predictions)
    # file_chunks = [chunk_names.count(file) for file in unique_files]
    # count = 0
    # for i in range(len(file_actuals)):
    #     if file_actuals[i] == file_predictions[i]:
    #         count += file_chunks[i]

    chunk_count_correct = {}
    chunk_count_total = {}


    for i in range(len(file_actuals)):
        if file_chunks[i] not in chunk_count_correct:
            chunk_count_total[file_chunks[i]] = 1
            if file_actuals[i] == file_predictions[i]:
                chunk_count_correct[file_chunks[i]] = 1
        else:
            chunk_count_total[file_chunks[i]] = chunk_count_total[file_chunks[i]] + 1
            chunk_count_correct[file_chunks[i]] = chunk_count_correct[file_chunks[i]] + (file_actuals[i] == file_predictions[i])

    # print(chunk_count_correct)
    # print(chunk_count_total)

    chunk_count_accuracy = {}

    for key in chunk_count_correct:
        chunk_count_accuracy[key] = 100*chunk_count_correct[key]/chunk_count_total[key]

    # print(chunk_count_accuracy)
    plt.bar(chunk_count_accuracy.keys(), chunk_count_accuracy.values())
    plt.show()
    plt.savefig(f"unsynced/pics/chunk_count_accuracy.png")

    cf_matrix=confusion_matrix(file_actuals, file_predictions)
    weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[1, 1])

    return weighted_accuracy


for train_dataset in ["physionet", "aisteth", "all"]:
    for test_data in ["physionet_2022", "AiSteth"]:
        print(train_dataset, test_data)
        for i in range(5):
            fold = f"fold_0{i+1}"
            technique = "whisper_small"
            model_path = f"unsynced/data/{technique}/results/{fold}_{train_dataset}/best_model.pt"
            print(model_path)

            model = load_downstream_model()

            input_dim = 768
            hidden_dim = 384
            num_classes = 2
            bidirectional = True
            num_layers = 2

            model= LSTMWithScaledDotProductAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)

            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)   

            # model = torch.load(model_path)
            model = model.to(device)


            test_dataset = MyDataset("unsynced/data/whisper_small", fold, [test_data], type="test")
            test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            print(fold, chunk_level_validation_accuracy(model, test_dataloader))
            # print(fold, file_level_accuracy(model, test_dataloader, fold, test_data))
        

