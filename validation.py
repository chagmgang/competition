import os
import json
import tqdm
import torch
import argparse
import torchvision
import transformers

import engine.bert
import model.bert
import dataset.dataset

import horovod.torch as hvd
import tensorboard.summary as tb_summary
import tensorflow as tf
import numpy as np
import pandas as pd

def main():

    config = json.load(open('config.json'))
    device = torch.device('cuda')
    validset = dataset.dataset.ValidationDataset(
        csv_file=config['test_file'],
        config=config)
    
    net = model.bert.Model(
            config=config).to(device)
    net.load_state_dict(torch.load('saved/albert_100000.pt'))
    net.eval()

    result = {}
    result['INDEX'] = list()
    result['NEWS1'] = list()
    result['NEWS2'] = list()

    for i in tqdm.tqdm(np.arange(len(validset))):

        input_ids, token_type_ids, attention_mask, index = validset[i]
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = net.prediction(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)

        labels = torch.argmax(logits, axis=1)
        labels = int(labels)

        if labels == 0:
            news1 = 0
            news2 = 1
        else:
            news1 = 1
            news2 = 0

        result['INDEX'].append(index)
        result['NEWS1'].append(news1)
        result['NEWS2'].append(news2)

        if i == 100:
            break

    dataframe = pd.DataFrame.from_dict(result)
    dataframe.to_csv('evaluation.csv', index=False)
    
if __name__ == '__main__':
    main()
