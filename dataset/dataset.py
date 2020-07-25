import tqdm
import copy
import json
import torch
import transformers
import torchvision

import numpy as np
import pandas as pd

def csv_to_dataframe(config, dataframe, arange, tokenizer):

    columns = copy.deepcopy(config['mask_field'])
    reverse_columns = copy.deepcopy(config['mask_field'])
    reverse_columns.reverse()
    mask_columns = copy.deepcopy(config['mask_column'])

    data = {}
    for mask_column in mask_columns:
        data[mask_column] = list()
    for mask_column in mask_columns:
        data[f'{mask_column}_index'] = list()
    data['LABEL'] = list()

    for i, idx in enumerate(tqdm.tqdm(arange)):

        single_data = dataframe.loc[idx]
        for x, y in zip(columns, mask_columns):
            data[y].append(single_data[x])
            token = tokenizer.tokenize(single_data[x])
            length = len(token)
            mask_arange = np.arange(length)
            np.random.shuffle(mask_arange)
            mask_arange = mask_arange[:int(length * config['mask_ratio'])]
            arange_string = ''
            for k in mask_arange:
                arange_string += f'{k},'
            data[f'{y}_index'].append(arange_string)
        data['LABEL'].append(0)

        for x, y in zip(reverse_columns, mask_columns):
            data[y].append(single_data[x])
            token = tokenizer.tokenize(single_data[x])
            length = len(token)
            mask_arange = np.arange(length)
            np.random.shuffle(mask_arange)
            mask_arange = mask_arange[:int(length * config['mask_ratio'])]
            arange_string = ''
            for k in mask_arange:
                arange_string += f'{k},'
            data[f'{y}_index'].append(arange_string)
        data['LABEL'].append(1)

    return pd.DataFrame.from_dict(data)

def make_mask_dataset(config):

    dataframe = pd.read_csv(config['csv_file'])

    arange = np.arange(len(dataframe))
    np.random.shuffle(arange)

    tokenizer = eval(config['tokenizer'])
    tokenizer = tokenizer.from_pretrained(config['tokenizer_path'])

    cut_index = int(len(arange) * 0.8)

    train_df = csv_to_dataframe(
            config=config,
            dataframe=dataframe,
            arange=arange[:cut_index],
            tokenizer=tokenizer)

    valid_df = csv_to_dataframe(
            config=config,
            dataframe=dataframe,
            arange=arange[cut_index:],
            tokenizer=tokenizer)

    train_df.to_csv(config['mask_train_file'], index=False)
    valid_df.to_csv(config['mask_valid_file'], index=False)

class MaskedLMDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, config):

        self.dataframe = pd.read_csv(csv_file)
        self.config = config
        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]

        seq_labels = data['LABEL']

        text = [data[f] for f in self.config['mask_column']]
        mask_indexs = [data[f'{f}_index'] for f in self.config['mask_column']]
        mask_indexs = self.regex(mask_indexs)

        tokens = [self.tokenizer.tokenize(t) for t in text]
        labels = copy.deepcopy(tokens)

        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        labels = [self.tokenizer.convert_tokens_to_ids(token) for token in labels]

        masked_ids = self.mask_ids(ids, mask_indexs)
        masked_ids, segment_ids, attention_mask = self.combine(masked_ids)
        labels, _, _ = self.combine(labels)

        masked_ids = self.padding(
                masked_ids,
                self.config['seq_length'],
                self.tokenizer.pad_token_id)
        segment_ids = self.padding(
                segment_ids,
                self.config['seq_length'],
                0)
        attention_mask = self.padding(
                attention_mask,
                self.config['seq_length'],
                0)
        labels = self.padding(
                labels,
                self.config['seq_length'],
                self.tokenizer.pad_token_id)

        masked_ids = torch.as_tensor(masked_ids, dtype=torch.long)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        seq_labels = torch.as_tensor(seq_labels, dtype=torch.long)

        return masked_ids, segment_ids, attention_mask, labels, seq_labels

    def padding(self, inputs, length, value):
        inputs = inputs[:length]

        while len(inputs) < length:
            inputs.append(value)

        return inputs

    def combine(self, ids):
        ids0 = ids[0]
        ids1 = ids[1]

        ids = self.tokenizer.build_inputs_with_special_tokens(
                token_ids_0=ids0,
                token_ids_1=ids1)

        segment = self.tokenizer.create_token_type_ids_from_sequences(
                token_ids_0=ids0,
                token_ids_1=ids1)

        attention_mask = [1] * len(ids)

        return ids, segment, attention_mask

    def mask_ids(self, tokens, mask_indexs):
        for token, mask_index in zip(tokens, mask_indexs):
            for index in mask_index:
                token[index] = self.tokenizer.mask_token_id
        return tokens

    def regex(self, index):
        result = []
        for i in index:
            i = i.split(',')[:-1]
            i = [int(ii) for ii in i]
            result.append(i)
        return result

    def __len__(self):
        return len(self.dataframe)

if __name__ == '__main__':
    config = json.load(open('config.json'))
    
    make_mask_dataset(
            config=config)
    
    dataset = MaskedLMDataset(
            csv_file=config['mask_valid_file'],
            config=config)

    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=8)

    for data in dataloader:
        for d in data:
            print(d.shape)
            print('-----')
