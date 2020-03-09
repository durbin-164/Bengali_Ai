import os
import ast
import torch
import numpy as np
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengliDatasetTrain
from torch import nn
from tqdm import tqdm
from early_stoping import EarlyStopping
from optimizers import Over9000
from utils import macro_recall

DEVICE = 'cuda'
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

BASE_MODEL = os.environ.get("BASE_MODEL")
BASE_LABEL = os.environ.get("BASE_LABEL")
OUTPUT_CHANNEL = int(os.environ.get("OUTPUT_CHANNEL"))


def loss_fn(output, target):

    loss = nn.CrossEntropyLoss()(output,target)

    return loss



def train(dataset, data_loader, model, optimizer, label_name):
    model.train()

    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter+=1
        image = d['image']
        target = d[label_name]
    
        image = image.to(DEVICE, dtype=torch.float)
        target = target.to(DEVICE, dtype = torch.long) 
        
        optimizer.zero_grad()

        output = model(image)
        
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        final_loss += loss

        
        final_outputs.append(output)
        final_targets.append(torch.unsqueeze(target,dim=1))
       
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets, label_name)
    
    return final_loss/counter , macro_recall_score


def evaluate(dataset, data_loader, model,optimizer, label_name):
    model.eval()
    final_loss = 0
    counter = 0
    final_loss = 0
    final_outputs = []
    final_targets = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
            counter = counter +1
            image = d['image']
            target = d[label_name]

            image = image.to(DEVICE, dtype=torch.float)
            target = target.to(DEVICE, dtype = torch.long)
            

            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)

            final_loss +=loss
            
            final_outputs.append(output)
            final_targets.append(torch.unsqueeze(target, dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================Evalutions=================")
        macro_recall_score = macro_recall(final_outputs, final_targets, label_name)
        

    return final_loss/counter,  macro_recall_score




def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, output_channel = OUTPUT_CHANNEL)
    model.to(DEVICE)

    train_dataset  = BengliDatasetTrain(
        folds = TRAINING_FOLDS,
        img_height = IMAGE_HEIGHT,
        img_width = IMAGE_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = 4
    )


    valid_dataset  = BengliDatasetTrain(
        folds = VALIDATION_FOLDS,
        img_height = IMAGE_HEIGHT,
        img_width = IMAGE_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False,
        num_workers = 4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    #optimizer =Over9000(model.parameters(), lr=2e-3, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                            patience = 0,factor=0.3, verbose=True)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    #base_dir = "Project/EducationProject/Bengali_Ai"
    model_name = "../save_model3/{}_{}_folds{}.bin".format(BASE_LABEL,BASE_MODEL, VALIDATION_FOLDS)

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    
    for epoch in range(EPOCHS):
        train_loss, train_score = train(train_dataset, train_loader, model, optimizer, BASE_LABEL)
        val_loss, val_score = evaluate(valid_dataset, valid_loader, model,optimizer, BASE_LABEL)
        scheduler.step(val_loss)

        early_stopping(val_loss, model, model_name)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #torch.save(model.state_dict(), f"{BASE_MODEL}_folds{VALIDATION_FOLDS}.bin")
    

if __name__ == "__main__":
    main()


