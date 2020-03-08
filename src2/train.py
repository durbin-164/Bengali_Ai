import os
import ast
import torch
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengliDatasetTrain
from torch import nn
from tqdm import tqdm
from early_stoping import EarlyStopping
#from optimizers import Over9000
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

    return (l1,l2,l3)



def train(dataset, data_loader, models, optimizers):

    
    counter = 0
    final_outputs = []
    final_targets = []

    g_final_loss = 0
    v_final_loss = 0
    c_final_loss = 0

    g_model, v_model, c_model = models
    g_optimizer, v_optimizer, c_optimizer = optimizers

    g_model.train()
    v_model.train()
    c_model.train()

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

        g_optimizer.zero_grad()
        v_optimizer.zero_grad()
        c_optimizer.zero_grad()

        g = g_model(image)
        v = v_model(image)
        c = c_model(image)

        outputs = (g,v,c)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        g_loss,v_loss,c_loss = loss_fn(outputs, targets)

        

        g_loss.backward()
        g_optimizer.step()

        v_loss.backward()
        v_optimizer.step()

        c_loss.backward()
        c_optimizer.step()

        g_final_loss += g_loss
        v_final_loss += v_loss
        c_final_loss += c_loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets

        final_outputs.append(torch.cat((o1,o2,o3), dim =1))
        final_targets.append(torch.stack((t1,t2,t3), dim =1))
    
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    final_losses = (g_final_loss/counter, v_final_loss/counter, c_final_loss/counter)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)
    
    return final_losses, macro_recall_score


def evaluate(dataset, data_loader, models,optimizers):
    counter = 0
    final_outputs = []
    final_targets = []

    g_final_loss = 0
    v_final_loss = 0
    c_final_loss = 0

    g_model, v_model, c_model = models
    g_optimizer, v_optimizer, c_optimizer = optimizers

    g_model.eval()
    v_model.eval()
    c_model.eval()

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

            g_optimizer.zero_grad()
            v_optimizer.zero_grad()
            c_optimizer.zero_grad()

            g = g_model(image)
            v = v_model(image)
            c = c_model(image)

            outputs = (g,v,c)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

            g_loss,v_loss,c_loss = loss_fn(outputs, targets)

            g_final_loss += g_loss
            v_final_loss += v_loss
            c_final_loss += c_loss

            o1, o2, o3 = outputs
            t1, t2, t3 = targets
            #print(t1.shape)
            final_outputs.append(torch.cat((o1,o2,o3), dim=1))
            final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)
        final_losses = (g_final_loss/counter, v_final_loss/counter, c_final_loss/counter)

        print("=================Evalutions=================")
        macro_recall_score = macro_recall(final_outputs, final_targets)
        

    return final_losses,  macro_recall_score




def main():
    g_model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, output_layer=168)
    g_model.to(DEVICE)
    v_model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True, output_layer=11)
    v_model.to(DEVICE)
    c_model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True,output_layer=7)
    c_model.to(DEVICE)

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

    #optimizer =Over9000(model.parameters(), lr=2e-3, weight_decay=1e-3)

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr = 1e-4)
    g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode="min", patience = 0,factor=0.3, verbose=True)

    v_optimizer = torch.optim.Adam(v_model.parameters(), lr = 1e-4)
    v_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(v_optimizer, mode="min", patience = 0,factor=0.3, verbose=True)

    c_optimizer = torch.optim.Adam(c_model.parameters(), lr = 1e-4)
    c_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(c_optimizer, mode="min", patience = 0,factor=0.3, verbose=True)


    g_early_stopping = EarlyStopping(patience=3, verbose=True)
    v_early_stopping = EarlyStopping(patience=3, verbose=True)
    c_early_stopping = EarlyStopping(patience=3, verbose=True)


    #base_dir = "Project/EducationProject/Bengali_Ai"
    g_model_name = "../save_model2/g_{}_folds{}.bin".format(BASE_MODEL, VALIDATION_FOLDS)
    v_model_name = "../save_model2/v_{}_folds{}.bin".format(BASE_MODEL, VALIDATION_FOLDS)
    c_model_name = "../save_model2/c_{}_folds{}.bin".format(BASE_MODEL, VALIDATION_FOLDS)

    if torch.cuda.device_count()>1:
        g_model = nn.DataParallel(g_model)
        v_model = nn.DataParallel(v_model)
        c_model = nn.DataParallel(c_model)

    
    models = (g_model, v_model, c_model)
    optimizers = (g_optimizer, v_optimizer, c_optimizer)
    schedulers = (g_scheduler, v_scheduler, c_scheduler)
    model_names = (g_model_name, v_model_name, c_model_name)
    
    for epoch in range(EPOCHS):
    
        train_losses, train_score = train(train_dataset, train_loader, models, optimizers)
        val_losses, val_score = evaluate(valid_dataset, valid_loader, models,optimizers)

        g_loss, v_loss, c_loss = val_losses

        g_scheduler.step(g_loss)
        v_scheduler.step(v_loss)
        c_scheduler.step(c_loss)

        g_early_stopping(g_loss, g_model, g_model_name)
        v_early_stopping(v_loss, v_model, v_model_name)
        c_early_stopping(c_loss, c_model, c_model_name)

        
        if g_early_stopping.early_stop and v_early_stopping and c_early_stopping:
            print(f"********Early stopping at epoches : {epoch}***********")
            break

        #torch.save(model.state_dict(), f"{BASE_MODEL}_folds{VALIDATION_FOLDS}.bin")
    

if __name__ == "__main__":
    main()


