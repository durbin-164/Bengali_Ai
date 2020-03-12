import os
import ast
import torch
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


def loss_fn(outputs, targets):
    o1,o2,o3 = outputs
    t1,t2,t3 = targets

    l1 = nn.CrossEntropyLoss()(o1,t1)
    l2 = nn.CrossEntropyLoss()(o2,t2)
    l3 = nn.CrossEntropyLoss()(o3,t3)

    return (l1+l2+l3)/3.0



def train(dataset, data_loader, model, optimizer):
    model.train()

    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter+=1
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

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets

        final_outputs.append(torch.cat((o1,o2,o3), dim =1))
        final_targets.append(torch.stack((t1,t2,t3), dim =1))
    
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)
    
    return final_loss/counter , macro_recall_score


def evaluate(dataset, data_loader, model,optimizer):
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


            o1, o2, o3 = outputs
            t1, t2, t3 = targets
            #print(t1.shape)
            final_outputs.append(torch.cat((o1,o2,o3), dim=1))
            final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================Evalutions=================")
        macro_recall_score = macro_recall(final_outputs, final_targets)
        

    return final_loss/counter,  macro_recall_score




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

    #optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    optimizer = torch.optim.AdamW(model.parameters())
    #optimizer =Over9000(model.parameters(), lr=2e-3, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                            patience = 0,factor=0.3, verbose=True)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    #base_dir = "Project/EducationProject/Bengali_Ai"
    model_name = "../save_model/{}_folds{}.bin".format(BASE_MODEL, VALIDATION_FOLDS)

    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    
    for epoch in range(EPOCHS):
        train_loss, train_score = train(train_dataset, train_loader, model, optimizer)
        val_loss, val_score = evaluate(valid_dataset, valid_loader, model,optimizer)
        scheduler.step(val_score)

        early_stopping(val_score, model, model_name)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        #torch.save(model.state_dict(), f"{BASE_MODEL}_folds{VALIDATION_FOLDS}.bin")
    

if __name__ == "__main__":
    main()


