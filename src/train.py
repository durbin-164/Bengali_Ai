import os
import ast
import torch
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengliDatasetTrain
from torch import nn
from tqdm import tqdm
from early_stoping import EarlyStopping

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


def loss_fn(outputs, targets):
    o1,o2,o3 = outputs
    t1,t2,t3 = targets

    l1 = nn.CrossEntropyLoss()(o1,t1)
    l2 = nn.CrossEntropyLoss()(o2,t2)
    l3 = nn.CrossEntropyLoss()(o3,t3)

    return (2*l1+l2+l3)/4.0



def train(dataset, data_loader, model, optimizer):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']


        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def evaluate(dataset, data_loader, model,optimizer):
    model.eval()
    final_loss = 0
    counter = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
            counter = counter +1
            image = d['image']
            grapheme_root = d['grapheme_root']
            vowel_diacritic = d['vowel_diacritic']
            consonant_diacritic = d['consonant_diacritic']


            image = image.to(DEVICE, dtype=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)

            optimizer.zero_grad()
            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)

            final_loss +=loss

    return final_loss/counter




def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                            patience = 1,factor=0.3, verbose=True)
    early_stopping = EarlyStopping(patience=7, verbose=True)

    #base_dir = "Project/EducationProject/Bengali_Ai"
    model_name = "../save_model/{}_folds{}.bin".format(BASE_MODEL, VALIDATION_FOLDS)

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    
    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        valid_loss = evaluate(valid_dataset, valid_loader, model,optimizer)
        scheduler.step(valid_loss)

        early_stopping(valid_loss, model, model_name)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #torch.save(model.state_dict(), f"{BASE_MODEL}_folds{VALIDATION_FOLDS}.bin")
    

if __name__ == "__main__":
    main()


