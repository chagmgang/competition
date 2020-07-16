import tqdm
import json
import torch
import transformers
import torchvision

import numpy as np
import pandas as pd

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, config):

        self.config = config
        self.dataframe = pd.read_csv(csv_file)
        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

        self.cls_ids = self.tokenizer.convert_tokens_to_ids([config['cls_token']])[0]
        self.pad_ids = self.tokenizer.convert_tokens_to_ids([config['pad_token']])[0]
        self.sep_ids = self.tokenizer.convert_tokens_to_ids([config['sep_token']])[0]

        self.cls_token = config['cls_token']
        self.pad_token = config['pad_token']
        self.sep_token = config['sep_token']

        self.length = config['seq_length']

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]
        
        before_text = [str(data[f]) for f in self.config['before_field']]
        after_text = [str(data[f]) for f in self.config['after_field']]

        before_input, before_segment, before_attention = self.combine(before_text)
        after_input, after_segment, after_attention = self.combine(after_text)

        before_input = torch.as_tensor(before_input, dtype=torch.long)
        before_segment = torch.as_tensor(before_segment, dtype=torch.long)
        before_attention = torch.as_tensor(before_attention, dtype=torch.long)

        after_input = torch.as_tensor(after_input, dtype=torch.long)
        after_segment = torch.as_tensor(after_segment, dtype=torch.long)
        after_attention = torch.as_tensor(after_attention, dtype=torch.long)

        label = data['LABEL']
        label = torch.as_tensor(label, dtype=torch.long)

        return before_input, before_segment, before_attention,\
                after_input, after_segment, after_attention, label

    def combine(self, text_list):
        tokenized_text = [self.tokenizer.tokenize(t) for t in text_list]
        
        tokens = [self.cls_token]
        segment_ids = [0]
        
        for i, t in enumerate(tokenized_text):
            segment = (i % 2)

            tokens.extend(t)
            tokens.append(self.sep_token)
            
            segment_ids.extend([segment for _ in range(len(t))])
            segment_ids.append(segment)

        attention_mask = [1 for _ in range(len(tokens))]

        tokens = tokens[:self.length]
        segment_ids = segment_ids[:self.length]
        attention_mask = attention_mask[:self.length]

        while len(tokens) < self.length:
            tokens.append(self.pad_token)
            segment_ids.append((i+1) % 2)
            attention_mask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return input_ids, segment_ids, attention_mask

    def __len__(self):
        return len(self.dataframe)

def split_data(csv_file, train_file, valid_file, ratio):

    original_df = pd.read_csv(csv_file)
    length = len(original_df)
    arange = np.arange(length)
    np.random.shuffle(arange)

    # train_idx = arange[:int(length * ratio)]
    train_idx = arange[:1000]
    valid_idx = arange[int(length * ratio):]

    train_data = {}
    train_data['INDEX'] = []
    train_data['NEWS1_HEADLINE'] = []
    train_data['NEWS1_BODY'] = []
    train_data['NEWS2_HEADLINE'] = []
    train_data['NEWS2_BODY'] = []
    train_data['LABEL'] = []
    for index, idx in enumerate(tqdm.tqdm(train_idx)):
        data = original_df.loc[idx]
        
        before_headline = data['BEFORE_HEADLINE']
        before_body    = data['BEFORE_BODY']
        after_headline = data['AFTER_HEADLINE']
        after_body     = data['AFTER_BODY']

        train_data['INDEX'].append(index * 2)
        train_data['NEWS1_HEADLINE'].append(before_headline)
        train_data['NEWS1_BODY'].append(before_body)
        train_data['NEWS2_HEADLINE'].append(after_headline)
        train_data['NEWS2_BODY'].append(after_body)
        train_data['LABEL'].append(0)

        train_data['INDEX'].append(index * 2 + 1)
        train_data['NEWS2_HEADLINE'].append(before_headline)
        train_data['NEWS2_BODY'].append(before_body)
        train_data['NEWS1_HEADLINE'].append(after_headline)
        train_data['NEWS1_BODY'].append(after_body)
        train_data['LABEL'].append(1)

    train_df = pd.DataFrame.from_dict(train_data)
    train_df.to_csv(train_file, index=False)

    valid_data = {}
    valid_data['INDEX'] = []
    valid_data['NEWS1_HEADLINE'] = []
    valid_data['NEWS1_BODY'] = []
    valid_data['NEWS2_HEADLINE'] = []
    valid_data['NEWS2_BODY'] = []
    valid_data['LABEL'] = []

    for index, idx in enumerate(tqdm.tqdm(valid_idx)):
        data = original_df.loc[idx]

        before_headline = data['BEFORE_HEADLINE']
        before_body    = data['BEFORE_BODY']
        after_headline = data['AFTER_HEADLINE']
        after_body     = data['AFTER_BODY']

        valid_data['INDEX'].append(index * 2)
        valid_data['NEWS1_HEADLINE'].append(before_headline)
        valid_data['NEWS1_BODY'].append(before_body)
        valid_data['NEWS2_HEADLINE'].append(after_headline)
        valid_data['NEWS2_BODY'].append(after_body)
        valid_data['LABEL'].append(0)

        valid_data['INDEX'].append(index * 2 + 1)
        valid_data['NEWS2_HEADLINE'].append(before_headline)
        valid_data['NEWS2_BODY'].append(before_body)
        valid_data['NEWS1_HEADLINE'].append(after_headline)
        valid_data['NEWS1_BODY'].append(after_body)
        valid_data['LABEL'].append(1)

    valid_df = pd.DataFrame.from_dict(valid_data)
    valid_df.to_csv(valid_file, index=False)

if __name__ == '__main__':

    config = json.load(open('config.json'))
    split_data(
            csv_file=config['csv_file'],
            train_file=config['train_file'],
            valid_file=config['valid_file'],
            ratio=0.8)

    '''
    config = json.load(open('config.json'))
    dataset = SequenceDataset(
            csv_file=config['train_file'],
            config=config)

    model = transformers.BertForNextSentencePrediction.from_pretrained(config['tokenizer_path'])
    model.eval()

    ## test 1
    text_list = ['Who was Jim Henson ?', 'Jim Henson was a puppeteer']
    input_ids, segment_ids, attention_mask = dataset.combine(text_list)
    input_tensors = torch.as_tensor([input_ids], dtype=torch.long)
    segment_tensors = torch.as_tensor([segment_ids], dtype=torch.long)
    attention_tensors = torch.as_tensor([attention_mask], dtype=torch.long)

    logit = model(
            input_ids=input_tensors,
            attention_mask=attention_tensors,
            token_type_ids=segment_tensors)
    print(logit)

    ### test 2
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = dataset.tokenizer.tokenize(text)
    indexed_tokens = dataset.tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    attention_mask = [1 for _ in range(len(segments_ids))]

    indexed_tokens = torch.as_tensor([indexed_tokens], dtype=torch.long)
    segments_ids = torch.as_tensor([segments_ids], dtype=torch.long)
    attention_mask = torch.as_tensor([attention_mask], dtype=torch.long)

    logit = model(
            input_ids=indexed_tokens,
            attention_mask=attention_mask,
            token_type_ids=segments_ids)
    print(logit)
    ### test 3

    text_list = ['Who was Jim Henson ?']
    input_ids, segment_ids, attention_mask = dataset.combine(text_list)
    input_tensors = torch.as_tensor([input_ids], dtype=torch.long)
    segment_tensors = torch.as_tensor([segment_ids], dtype=torch.long)
    attention_tensors = torch.as_tensor([attention_mask], dtype=torch.long)

    logit = model(
            input_ids=input_tensors,
            attention_mask=attention_tensors,
            token_type_ids=segment_tensors)
    print(logit)

    ### test 4
    text = "[CLS] Who was Jim Henson ? [SEP]"
    tokenized_text = dataset.tokenizer.tokenize(text)
    indexed_tokens = dataset.tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0]
    attention_mask = [1 for _ in range(len(segments_ids))]

    indexed_tokens = torch.as_tensor([indexed_tokens], dtype=torch.long)
    segments_ids = torch.as_tensor([segments_ids], dtype=torch.long)
    attention_mask = torch.as_tensor([attention_mask], dtype=torch.long)

    logit = model(
            input_ids=indexed_tokens,
            attention_mask=attention_mask,
            token_type_ids=segments_ids)
    print(logit)
    '''
