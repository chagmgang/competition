import tqdm
import copy
import json
import torch
import transformers
import torchvision

import numpy as np
import pandas as pd

class MaskedLMDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, config):

        self.dataframe = pd.read_csv(csv_file)
        self.config = config
        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]
        
        seq1 = data['SEQ1']
        seq2 = data['SEQ2']
        seq1_mask_index = self.regex(data['SEQ1_INDEX'])
        seq2_mask_index = self.regex(data['SEQ2_INDEX'])
        
        token1 = self.tokenizer.tokenize(seq1)
        token2 = self.tokenizer.tokenize(seq2)

        mask_token1 = copy.deepcopy(token1)
        mask_token2 = copy.deepcopy(token2)

        mask_token1 = self.mask(
                token=mask_token1,
                idx=seq1_mask_index)
        mask_token2 = self.mask(
                token=mask_token2,
                idx=seq2_mask_index)

        labels, _, _ = self.combine(
                token1=token1,
                token2=token2)
        input_ids, token_type_ids, attention_mask = self.combine(
                token1=mask_token1,
                token2=mask_token2)

        seq_labels = data['LABEL']

        input_ids = self.padding(
                inputs=input_ids,
                length=self.config['seq_length'],
                value=0)
        token_type_ids = self.padding(
                inputs=token_type_ids,
                length=self.config['seq_length'],
                value=0)
        attention_mask = self.padding(
                inputs=attention_mask,
                length=self.config['seq_length'],
                value=0)
        labels = self.padding(
                inputs=labels,
                length=self.config['seq_length'],
                value=0)

        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.as_tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        seq_labels = torch.as_tensor(seq_labels, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels, seq_labels

    def combine(self, token1, token2):
        ids1 = self.tokenizer.convert_tokens_to_ids(token1)
        ids2 = self.tokenizer.convert_tokens_to_ids(token2)

        ids = self.tokenizer.build_inputs_with_special_tokens(
                token_ids_0=ids1,
                token_ids_1=ids2)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                token_ids_0=ids1,
                token_ids_1=ids2)
        attention_mask = [1] * len(ids)

        return ids, token_type_ids, attention_mask

    def mask(self, token, idx):
        for i in idx:
            token[i] = self.tokenizer.mask_token
        return token

    def padding(self, inputs, length, value):
        inputs = inputs[:length]

        while len(inputs) < length:
            inputs.append(value)

        return inputs

    def regex(self, index):
        index = index.split(',')[:-1]
        index = [int(ind) for ind in index]
        return index

    def __len__(self):
        return len(self.dataframe)

class MakingDataset:

    def __init__(self, config):
        
        self.config = config
        self.ratio = 0.8

        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

        self.dataframe = pd.read_csv(config['csv_file'])
        
        train_idx, valid_idx = self.make_index()

        train_frame = self.make_empty_dataframe()
        valid_frame = self.make_empty_dataframe()
        
        train_dataframe = self.make_dataset(
                dataframe=self.dataframe,
                idx=train_idx,
                frame=train_frame,
                repeat=5)

        valid_dataframe = self.make_dataset(
                dataframe=self.dataframe,
                idx=valid_idx,
                frame=valid_frame,
                repeat=1)

        train_dataframe.to_csv(self.config['mask_train_file'])
        valid_dataframe.to_csv(self.config['mask_valid_file'])

    def make_dataset(self, dataframe, idx, frame, repeat):

        step = 0
        for _, i in enumerate(tqdm.tqdm(idx)):

            single_data = dataframe.loc[i]

            for _ in range(repeat):
                
                true_data = self.make_data(
                        single_data=single_data,
                        index=step,
                        reverse=False)
                
                step += 1
                
                false_data = self.make_data(
                        single_data=single_data,
                        index=step,
                        reverse=True)
                
                step += 1
                
                for k in true_data.keys():
                    frame[k].append(true_data[k])
                for k in false_data.keys():
                    frame[k].append(false_data[k])

        frame = pd.DataFrame.from_dict(frame)
        return frame

    def make_data(self, single_data, index, reverse):

        if not reverse:
            seq1_headline = str(single_data['BEFORE_HEADLINE'])
            seq1_body     = str(single_data['BEFORE_BODY'])
            seq2_headline = str(single_data['AFTER_HEADLINE'])
            seq2_body     = str(single_data['AFTER_BODY'])
            label         = 0   ###### -> before & after

        else:
            seq2_headline = str(single_data['BEFORE_HEADLINE'])
            seq2_body     = str(single_data['BEFORE_BODY'])
            seq1_headline = str(single_data['AFTER_HEADLINE'])
            seq1_body     = str(single_data['AFTER_BODY'])
            label         = 1   ###### -> after & before

        if str(seq1_headline) == 'nan':
            seq1_headline = ''
        if str(seq1_body) == 'nan':
            seq1_body = ''
        if str(seq2_headline) == 'nan':
            seq2_headline = ''
        if str(seq2_body) == 'nan':
            seq2_body = ''

        seq1 = ' '.join([seq1_headline, seq1_body])
        seq2 = ' '.join([seq2_headline, seq2_body])

        seq1 = seq1_body
        seq2 = seq2_body

        seq1_mask_index = self.make_mask_index(
                text=seq1, ratio=0.15)
        seq2_mask_index = self.make_mask_index(
                text=seq2, ratio=0.15)

        data = dict()
        data['INDEX'] = index
        data['SEQ1'] = seq1
        data['SEQ2'] = seq2
        data['SEQ1_INDEX'] = seq1_mask_index
        data['SEQ2_INDEX'] = seq2_mask_index
        data['LABEL'] = label
        
        return data

    def make_mask_index(self, text, ratio):
        
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        arange = np.arange(length)
        np.random.shuffle(arange)
        arange = arange[:int(length * ratio)]
        arange_string = ''
        for aran in arange:
            arange_string += f'{aran},'
        return arange_string

    def make_empty_dataframe(self):
        data = dict()
        data['INDEX'] = list()
        data['SEQ1'] = list()
        data['SEQ2'] = list()
        data['SEQ1_INDEX'] = list()
        data['SEQ2_INDEX'] = list()
        
        data['LABEL'] = list()
        return data

    def make_index(self):
        data_length = len(self.dataframe)
        arange = np.arange(data_length)
        np.random.shuffle(arange)
        cut_index = int(data_length * self.ratio)
        train_idx = arange[:cut_index]
        valid_idx = arange[cut_index:]

        return train_idx, valid_idx

class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, config):

        self.dataframe = pd.read_csv(csv_file)
        self.config = config
        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]

        index = data['INDEX']
        news1_headline = data['NEWS1_HEADLINE']
        news1_body = data['NEWS1_BODY']
        news2_headline = data['NEWS2_HEADLINE']
        news2_body = data['NEWS2_BODY']

        news1 = self.make_seq(
                headline=news1_headline,
                body=news1_body)
        news2 = self.make_seq(
                headline=news2_headline,
                body=news2_body)

        token1 = self.tokenizer.tokenize(news1)
        token2 = self.tokenizer.tokenize(news2)
        input_ids, token_type_ids, attention_mask = self.combine(
                token1=token1,
                token2=token2)

        input_ids = self.padding(
                inputs=input_ids,
                length=self.config['seq_length'],
                value=0)
        token_type_ids = self.padding(
                inputs=token_type_ids,
                length=self.config['seq_length'],
                value=0)
        attention_mask = self.padding(
                inputs=attention_mask,
                length=self.config['seq_length'],
                value=0)

        print(f'news1 : {news1}')
        print(f'news2 : {news2}')

        input_ids = torch.as_tensor([input_ids], dtype=torch.long)
        token_type_ids = torch.as_tensor([token_type_ids], dtype=torch.long)
        attention_mask = torch.as_tensor([attention_mask], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, index

    def padding(self, inputs, length, value):
        inputs = inputs[:length]
        while len(inputs) < length:
            inputs.append(value)
        return inputs

    def combine(self, token1, token2):
        ids1 = self.tokenizer.convert_tokens_to_ids(token1)
        ids2 = self.tokenizer.convert_tokens_to_ids(token2)

        ids = self.tokenizer.build_inputs_with_special_tokens(
                token_ids_0=ids1,
                token_ids_1=ids2)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                token_ids_0=ids1,
                token_ids_1=ids2)
        attention_mask = [1] * len(ids)
        return ids, token_type_ids, attention_mask

    def make_seq(self, headline, body):
        if str(headline) == 'nan':
            headline = ''
        if str(body) == 'nan':
            body = ''

        seq = ' '.join([headline, body])
        return seq

    def __len__(self):
        return len(self.dataframe)


if __name__ == '__main__':
    config = json.load(open('config.json'))
    MakingDataset(
            config=config)
    
    '''
    dataset = MaskedLMDataset(
            csv_file=config['mask_train_file'],
            config=config)    
    data = dataset[0]
    for d in data:
        print(d)
    data = dataset[1]
    print('----')
    for d in data:
        print(d)
    
    dataset = MaskedLMDataset(
            csv_file=config['mask_valid_file'],
            config=config)
    data = dataset[0]
    for d in data:
        print(d)
    data = dataset[1]
    print('----')
    for d in data:
        print(d)
    '''
