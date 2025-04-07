from utils.lstm_additive_satt import LSTMWithAdditiveSelfAttention
from utils.lstm_scaled_attn import LSTMWithScaledDotProductAttention
from utils.wav2vec2_ll import Wav2Vec2LL
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os, glob, torch, logging
import json
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import torch.nn.functional as F

def save_plots(save_path, train_accuracy, acc_legend, train_losses, loss_legend, test_accuracy, test_acc_legend, test_losses, test_loss_legend):
    plt.subplot(121)
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend([acc_legend, test_acc_legend])
    plt.ylim(0, 110)
    
    plt.subplot(122)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend([loss_legend, test_loss_legend])

    save_plot_path=os.path.join(save_path, 'result.png')
    plt.savefig(save_plot_path)
    plt.close()
    return

# def calc_wt_accuracy(cf_matrix, weights=[5, 1, 3]):
#     a=cf_matrix
#     nlabels=a.shape[0]
#     numerator=0
#     denominator=0
#     for i in range(nlabels):
#         numerator+=weights[i]*a[i,i]
#         denominator+=weights[i]*a[i].sum()
#     return numerator/denominator

def calc_wt_accuracy(cf_matrix, weights=[5, 1]):
    a=cf_matrix
    nlabels=a.shape[0]
    numerator=0
    denominator=0
    for i in range(nlabels):
        numerator+=weights[i]*a[i,i]
        denominator+=weights[i]*a[i].sum()
    return 100*numerator/denominator



# -------------------------------------------------------------------------
def train_test_evaluate(model, logger, trial_folder_path, train_dataloader, test_dataloader, batch_size, learning_rate, num_epochs, device, scheduler_dct, optimizer, class_weights, classifier_type="LSTM"):

    inputs, labels=next(iter(train_dataloader))
    logger.info(f"inputs.shape: {inputs.shape},\tlabels.shape: {labels.shape}")

    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    logger.info(f"Class weights: {class_weights}")\
        
    # print(classifier_type)

    if classifier_type=="LSTM":
        criterion = nn.CrossEntropyLoss(weight=None)
    elif classifier_type=="LinearLayer":
        # criterion = nn.CTCLoss(blank=2, zero_infinity=True)
        criterion = nn.CrossEntropyLoss(weight=None)

    if optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Unexpected optimizer found. Expected one of the (adam,). Using default adam as of now.")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dct)


    # Train loop
    train_losses=[]
    test_losses=[]
    train_accuracies=[]
    test_accuracies=[]
    weighted_accuracies = []

    # --------------------------------------------------------------------
    best_model_path=None
    best_weighted_accuracy=0

    # Loops
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}".center(120, '-') )
        logger.info(f'Epoch: {epoch+1}'.center(120, '-'))

        # ---------------Training Mode-----------------------------
        model.train()
        running_loss = 0.0
        running_accuracies=[]
        c = 0
        for inputs, labels in tqdm(train_dataloader):
            inputs,labels=inputs.to(device), labels.to(device)
            # print(type(inputs), type(labels))

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            if classifier_type=="LSTM":
                # print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
            elif classifier_type=="LinearLayer":
                # # Ensure labels are a 1D tensor for CTC Loss
                # if labels.dim() != 1:
                #     raise ValueError("labels must be a 1D tensor for CTC Loss")
                
                # log_probs = outputs.transpose(0, 1)  # (T, N, C)
                # log_probs = F.log_softmax(log_probs, dim=-1)
                # T, N, C = log_probs.shape
                # input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
                # target_lengths = torch.tensor([len(labels) // N] * N, dtype=torch.long)
                
                # loss = criterion(log_probs, labels, input_lengths, target_lengths)
                # log_probs = F.log_softmax(outputs, dim=2)

                # print(outputs.shape, labels.shape)
                # print(outputs.shape, labels.shape)
                # convert output into the labels
                # log_probs = F.log_softmax(outputs, dim=-1)
                # print(log_probs.shape, labels.shape)
                loss = criterion(outputs, labels)


            # #print model weights
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)



            if torch.isnan(loss):
                logger.warn('nan')
                continue

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            accuracy= (outputs.argmax(dim=-1)==labels).float().mean().item()
            running_accuracies.append(accuracy)
        
        # Calculate average training loss for the epoch
        average_train_loss = running_loss / len(train_dataloader)
        average_train_accuracy=sum(running_accuracies)/len(running_accuracies)*100
        train_accuracies.append(average_train_accuracy)
        train_losses.append(average_train_loss)
        logger.info(f"Train Loss: {average_train_loss},\tTrain Accuracy: {average_train_accuracy:.2f}%")
        
        # ----------------------------Validation Mode--------------------------------------
        model.eval()
        val_running_accuracies=[]
        running_loss=0
        val_actuals=[]
        val_predictions=[]
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(test_dataloader):
                val_inputs, val_labels=val_inputs.to(device), val_labels.to(device)
                if val_inputs.shape[0]!=batch_size:
                    print("Error. Testing input is not same as batch size.")

                val_outputs = model(val_inputs)
                # val_loss = criterion(val_outputs, val_labels)
                if classifier_type=="LSTM":
                    # print(val_outputs, val_labels)
                    val_loss = criterion(val_outputs, val_labels)
                elif classifier_type=="LinearLayer":
                    # log_probs = F.log_softmax(val_outputs, dim=2)
                    # log_probs = log_probs.transpose(0,1)
                    # input_lengths = torch.full(size=(val_outputs.size(0),), fill_value=1, dtype=torch.long)
                    # target_lengths = torch.full(size=(val_outputs.size(0),), fill_value=1, dtype=torch.long)
                    # target_labels = val_labels.clone().detach()
                    # val_loss = criterion(log_probs, target_labels, input_lengths, target_lengths)
                    # val_outputs = log_probs.mean(dim=0)

                    # log_probs = F.log_softmax(val_outputs, dim=-1)
                    # print(log_probs.shape, labels.shape)

                    # print(val_outputs, val_labels)

                    val_loss = criterion(val_outputs, val_labels)

                val_actuals.extend(val_labels.detach().cpu().numpy().astype(int).tolist())
                val_predictions.extend(val_outputs.argmax(dim=-1).detach().cpu().numpy().tolist())

                if torch.isnan(val_loss):
                    logger.warn('nan')
                    continue

                running_loss += val_loss.item()
                val_accuracy= (val_outputs.argmax(dim=-1)==val_labels).float().mean().item()
                val_running_accuracies.append(val_accuracy)

        # Calculate for evaluation epoch
        average_test_loss = running_loss / len(test_dataloader)
        average_test_accuracy=sum(val_running_accuracies)/len(val_running_accuracies)*100
            
        test_accuracies.append(average_test_accuracy)
        test_losses.append(average_test_loss)
        logger.info(f"Test Loss: {average_test_loss},\tTest Accuracy: {average_test_accuracy:.2f}%")
        
        cf_matrix=confusion_matrix(val_actuals, val_predictions)
        weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=[5, 1, 1])
        # weighted_accuracy=calc_wt_accuracy(cf_matrix, weights=class_weights)
        weighted_accuracies.append(weighted_accuracy)

        logger.info(f"Weighted accuracy: {weighted_accuracy}")
        logger.info(f"Test: confusion_matrix:\n{cf_matrix}")
        logger.info(f"Test: classification report:\n{classification_report(val_actuals, val_predictions)}")

        if scheduler_dct['mode']=="max":
            scheduler.step(average_test_accuracy)
        elif scheduler_dct['mode']=="min":
            scheduler.step(average_test_loss)
        else:
            print("Unexpected input received for scheduler. Needs one of (min, max) else scheduler will not be used.")
        # Saving the plot
        save_plots(trial_folder_path, train_accuracies, 'train_accuracy', train_losses, 'train_loss', test_accuracies, 'test_accuracy', test_losses, 'test_loss')
        df=pd.DataFrame({"epoch num":list(range(1, len(train_accuracies)+1)),
                        "train accuracy":train_accuracies, "test accuracy":test_accuracies, 
                        "train loss":train_losses, "test loss":test_losses, "weighted accuracy":weighted_accuracies})
        df.to_csv(f"{trial_folder_path}/results.csv", index=False)
        
        if weighted_accuracy>best_weighted_accuracy:
            best_model_path=f"{trial_folder_path}/best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            best_weighted_accuracy=weighted_accuracy
            logger.info("best model saved.")

    logger.info(f"Best Weighted Accuracy: {best_weighted_accuracy}")


# # -----------------------------------------------------------------------------------
def create_logger(trial_folder_path):   
    log_filepath=os.path.join(trial_folder_path, "logs.log")
    logging.basicConfig(filename=log_filepath, format='%(asctime)s : %(message)s', filemode='w', level=logging.INFO)
    print("Find logs -->", log_filepath)
    return logging.getLogger()

def load_downstream_model(model_type="LSTMWithScaledDotProductAttention", input_dim=100, hidden_dim=64, num_classes=2, bidirectional=True, num_layers=3):
    if model_type=="LSTMWithScaledDotProductAttention":
        model = LSTMWithScaledDotProductAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)
    elif model_type=="LSTMWithAdditiveSelfAttention":
        model = LSTMWithAdditiveSelfAttention(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)
    else:
        raise ValueError("Unexpected model type received. Expected one of (LSTMWithScaledDotProductAttention, LSTMWithAdditiveSelfAttention)")
    return model

class MyDataset(Dataset):
    def __init__(self, folder_path, fold, datasets, type):
        self.filepaths=[]
        for dataset in datasets:
            self.filepaths.extend(glob.glob(f"{folder_path}/{fold}/{dataset}/{type}/*/*.pt"))
        print(len(self.filepaths))
        self.labels=[int(filepath.rsplit("/", 2)[-2]) for filepath in self.filepaths]

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        y=torch.load(self.filepaths[index], map_location='cuda:5').squeeze(0)
        label=self.labels[index]
        return y, label
    
class LoadDataset(Dataset):
    def __init__(self, folder_path, fold, datasets, type):
        self.filepaths=[]
        self.labels=[]
        for dataset in datasets:
            csv_file = f"{folder_path}/{fold}/{dataset}/train-test-files.csv"
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                if df.iloc[i]["set_type"]==type:
                    self.filepaths.append(f"{df.iloc[i]['filepath']}")
                    self.labels.append(int(df.iloc[i]["label"]))
        print(len(self.filepaths))

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        #load the audio file
        # y=torch.load(self.filepaths[index]).squeeze(0)
        y, sr=librosa.load(self.filepaths[index], sr=None)
        label=self.labels[index]
        # print(y.shape, label)
        # print(y)
        return y, label
    

def perform_downstream_task(model_training_config, device, fold):
    trial_folder_path = model_training_config["dir_paths"]["trial_folder_path"]
    batch_size = model_training_config["training_params"]["batch_size"]                          
    learning_rate = model_training_config["training_params"]["learning_rate"]                     
    num_epochs = model_training_config["training_params"]["num_epochs"]                     
    scheduler_dct = model_training_config["training_params"]["scheduler"]                               
    optimizer = model_training_config["training_params"]["optimizer"]                            
    model_type = model_training_config["architecture"]["model_type"]    
    datasets = model_training_config["datasets"]                          

    input_dim=model_training_config["architecture"]["input_dim"]
    hidden_dim=model_training_config["architecture"]["hidden_dim"]
    num_classes=model_training_config["architecture"]["num_classes"]
    bidirectional=model_training_config["architecture"]["bidirectional"]
    num_layers=model_training_config["architecture"]["num_layers"]

    os.makedirs(trial_folder_path, exist_ok=False)
    logger = create_logger(trial_folder_path=trial_folder_path)
    model=load_downstream_model(model_type = model_type, input_dim = input_dim, hidden_dim = hidden_dim, num_classes=num_classes, bidirectional = bidirectional, num_layers = num_layers)
    # model = torch.load("unsynced/data/base_model_1/results/fold_05_normal_all/best_model.pt")
    model.to(device)

    datasets_path = os.path.join(trial_folder_path, "../..")


    print("Datasets: ", datasets)

    train_dataset = MyDataset(datasets_path, fold, datasets, type="train")
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = MyDataset(datasets_path, fold, datasets, type="test")
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    normal_chunks_count = train_dataset.labels.count(1) + test_dataset.labels.count(1)
    murmur_chunks_count = train_dataset.labels.count(0) + test_dataset.labels.count(0)
    class_weights = [round(normal_chunks_count/murmur_chunks_count), 1]
    
    train_test_evaluate(model=model, logger=logger, trial_folder_path=trial_folder_path, train_dataloader=train_dataloader, test_dataloader=test_dataloader, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, device=device, scheduler_dct=scheduler_dct, optimizer=optimizer, class_weights=class_weights)


def perform_downstream_task_with_model(model, model_training_config, device, fold):
    trial_folder_path = model_training_config["dir_paths"]["trial_folder_path"]
    batch_size = model_training_config["training_params"]["batch_size"]                          
    learning_rate = model_training_config["training_params"]["learning_rate"]                     
    num_epochs = model_training_config["training_params"]["num_epochs"]                     
    scheduler_dct = model_training_config["training_params"]["scheduler"]                               
    optimizer = model_training_config["training_params"]["optimizer"]                            
    model_type = model_training_config["architecture"]["model_type"]    
    datasets = model_training_config["datasets"]                          

    input_dim=model_training_config["architecture"]["input_dim"]
    hidden_dim=model_training_config["architecture"]["hidden_dim"]
    num_classes=model_training_config["architecture"]["num_classes"]
    bidirectional=model_training_config["architecture"]["bidirectional"]
    num_layers=model_training_config["architecture"]["num_layers"]

    os.makedirs(trial_folder_path, exist_ok=False)
    # os.makedirs(trial_folder_path, exist_ok=True)
    logger = create_logger(trial_folder_path=trial_folder_path)
    # model=load_downstream_model(model_type = model_type, input_dim = input_dim, hidden_dim = hidden_dim, num_classes=num_classes, bidirectional = bidirectional, num_layers = num_layers)

    # model = torch.load("unsynced/data/base_model_1/results/fold_05_normal_all/best_model.pt")
    model = Wav2Vec2LL(model, num_classes=num_classes)
    model.to(device)

    print(model)

    datasets_path = os.path.join(trial_folder_path, "../../..")


    print("Datasets: ", datasets)

    train_dataset = LoadDataset(datasets_path, fold, datasets, type="train")
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = LoadDataset(datasets_path, fold, datasets, type="test")
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    normal_chunks_count = train_dataset.labels.count(1) + test_dataset.labels.count(1)
    murmur_chunks_count = train_dataset.labels.count(0) + test_dataset.labels.count(0)
    class_weights = [round(normal_chunks_count/murmur_chunks_count), 1]
    
    train_test_evaluate(model=model, logger=logger, trial_folder_path=trial_folder_path, train_dataloader=train_dataloader, test_dataloader=test_dataloader, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, device=device, scheduler_dct=scheduler_dct, optimizer=optimizer, class_weights=class_weights)



# def perform_downstream_task_with_model(model, model_training_config, device, fold):
#     trial_folder_path = model_training_config["dir_paths"]["trial_folder_path"]
#     batch_size = model_training_config["training_params"]["batch_size"]                          
#     learning_rate = model_training_config["training_params"]["learning_rate"]                     
#     num_epochs = model_training_config["training_params"]["num_epochs"]                     
#     scheduler_dct = model_training_config["training_params"]["scheduler"]                               
#     optimizer = model_training_config["training_params"]["optimizer"]   
#     datasets = model_training_config["datasets"]                      
#     num_classes=model_training_config["architecture"]["num_classes"]

#     # os.makedirs(trial_folder_path, exist_ok=False)
#     os.makedirs(trial_folder_path, exist_ok=True)
#     logger = create_logger(trial_folder_path=trial_folder_path)
#     model = Wav2Vec2LL(model, num_classes=num_classes)
#     # print(model)
#     model.to(device)

#     datasets_path = os.path.join(trial_folder_path, "../../..")

#     print("Datasets: ", datasets)

#     train_dataset = LoadDataset(datasets_path, fold, datasets, type="train")
#     train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_dataset = LoadDataset(datasets_path, fold, datasets, type="test")
#     test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#     normal_chunks_count = train_dataset.labels.count(1) + test_dataset.labels.count(1)
#     normal_chunks_count = train_dataset.labels.count(1) + test_dataset.labels.count(1)
#     murmur_chunks_count = train_dataset.labels.count(0) + test_dataset.labels.count(0)
#     class_weights = [round(normal_chunks_count/murmur_chunks_count), 1]
    
#     train_test_evaluate(model=model, logger=logger, trial_folder_path=trial_folder_path, train_dataloader=train_dataloader, test_dataloader=test_dataloader, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, device=device, scheduler_dct=scheduler_dct, optimizer=optimizer, class_weights=class_weights, classifier_type="LinearLayer")
    