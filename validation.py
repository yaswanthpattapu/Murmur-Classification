import torch
from torch.utils.data import DataLoader
from train_test_evaluate import MyDataset, calc_wt_accuracy
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

batch_size = 32
device = "cuda:1" if torch.cuda.is_available() else "cpu"

validation_dataset = MyDataset("unsynced/data/base_model_1", "validation", ["AiSteth"], type="validation")
validation_dataloader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

def calculte_val_acc_all_models():
    for i in range(5):
        fold = f"fold_0{i+1}"
        for ds in ["physionet", "aisteth", "all"]:
            model_name = f"{fold}_normal_{ds}"
            model = f"unsynced/data/base_model_1/results/{model_name}/best_model.pt"
            model = torch.load(model)
            model = model.to(device)

            print(f"{fold} - {ds} - {chunk_level_validation_accuracy(model)}")

def get_labels(model, validation_dataloader):
    criterion = nn.CrossEntropyLoss(weight=None)

    model.eval()
    val_running_accuracies=[]
    running_loss=0
    val_actuals=[]
    val_predictions=[]

    with torch.no_grad():
        # for val_inputs, val_labels in tqdm(validation_dataloader):
        for val_inputs, val_labels in tqdm(validation_dataloader):
            val_inputs, val_labels=val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_actuals.extend(val_labels.detach().cpu().numpy().astype(int).tolist())
            val_predictions.extend(val_outputs.argmax(dim=-1).detach().cpu().numpy().tolist())

            if torch.isnan(val_loss):
                continue

            running_loss += val_loss.item()
            val_accuracy= (val_outputs.argmax(dim=-1)==val_labels).float().mean().item()
            val_running_accuracies.append(val_accuracy)

    return val_actuals ,val_predictions
    

def chunk_level_validation_accuracy(model):
    val_actuals ,val_predictions = get_labels(model, validation_dataloader)

    cf_matrix=confusion_matrix(val_actuals, val_predictions)
    weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[2, 1])

    return weighted_accuracy

def file_level_validation_accuracy(model):
    df = pd.read_csv(f"unsynced/data/validation/AiSteth/validation-files.csv")

    file_names = list(df["filename"])
    file_actuals = list(df["label"])

    uniques_files = list(set(file_names))
    uniques_files_labels = df.groupby("filename")["label"].unique()
    file_chunks = df.groupby("filename")["filename"].count().mean()
    print(len(uniques_files), len(uniques_files_labels))
    print(file_chunks)

    val_actuals ,val_predictions = get_labels(model, validation_dataloader)

    

    for file in uniques_files:
        file_actual = []
        file_prediction = []    
        for i in range(len(file_names)):
            if file_names[i] == file:
                file_actual.append(file_actuals[i])
                file_prediction.append(val_predictions[i])

        cf_matrix=confusion_matrix(file_actual, file_prediction)
        weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[2, 1])

        print(f"File: {file} - Weighted Accuracy: {weighted_accuracy}")

    cf_matrix=confusion_matrix(val_actuals, val_predictions)
    weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[2, 1])

    return weighted_accuracy

if __name__ == "__main__":
    calculte_val_acc_all_models()
    # print(file_level_validation_accuracy())