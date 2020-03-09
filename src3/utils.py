import numpy as np
import pandas as pd
import torch
import sklearn
import sklearn.metrics

def macro_recall(pred_y, y, label_name):
    
    
    #pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]
    pred_labels = torch.argmax(pred_y, axis = 1).cpu().numpy()

    y = y.cpu().numpy()

    recall = sklearn.metrics.recall_score(pred_labels, y[:, 0], average='macro')
    
    print(f'recall: {label_name} ->  {recall}, y {y.shape}')
    
    return recall