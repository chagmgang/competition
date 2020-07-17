import tqdm
import json
import torch
import transformers
import torchvision

import numpy as np
import pandas as pd

class SequenceDatasetValid(torch.utils.data.Dataset):

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
        self.mask_token = config['mask_token']

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

class SequenceDatasetTrain(torch.utils.data.Dataset):

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
        self.mask_token = config['mask_token']

        self.length = config['seq_length']

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]

        before_text = [str(data[f]) for f in self.config['before_field']]
        after_text = [str(data[f]) for f in self.config['after_field']]

        before_masking_index = data['NEWS1_MASKING_INDEX']
        after_masking_index = data['NEWS2_MASKING_INDEX']

        before_masking_index = before_masking_index.split(',')[:-1]
        after_masking_index = after_masking_index.split(',')[:-1]

        before_masking_index = [int(i) for i in before_masking_index]
        after_masking_index = [int(i) for i in after_masking_index]

        before_input, before_segment, before_attention = self.combine(
                text_list=before_text,
                mask_text=1,
                mask_index=before_masking_index)

        after_input, after_segment, after_attention = self.combine(
                text_list=after_text,
                mask_text=1,
                mask_index=after_masking_index)

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

    def combine(self, text_list, mask_text, mask_index):
        tokenized_text = [self.tokenizer.tokenize(t) for t in text_list]

        tokens = [self.cls_token]
        segment_ids = [0]

        for i, t in enumerate(tokenized_text):

            segment = (i % 2)

            if i == mask_text:
                for j in mask_index:
                    t[j] = self.mask_token

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

class SplitData:

    def __init__(self, config, ratio, repeat):

        self.config = config
        self.repeat = repeat
        self.ratio = ratio

        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

        self.df = pd.read_csv(config['csv_file'])
        self.df_length = len(self.df)
        self.ratio = ratio

        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

        self.df = pd.read_csv(config['csv_file'])
        self.df_length = len(self.df)
        self.arange = np.arange(self.df_length)
        np.random.shuffle(self.arange)

        self.train_idx = self.arange[:int(self.df_length * ratio)]
        self.valid_idx = self.arange[int(self.df_length * ratio):]

        
        train_data = {}
        train_data['INDEX'] = []
        train_data['NEWS1_HEADLINE'] = []
        train_data['NEWS1_BODY'] = []
        train_data['NEWS1_MASKING_INDEX'] = []
        train_data['NEWS2_HEADLINE'] = []
        train_data['NEWS2_BODY'] = []
        train_data['NEWS2_MASKING_INDEX'] = []
        train_data['LABEL'] = []

        train_step = 0
        for idx in tqdm.tqdm(self.train_idx):

            data = self.df.loc[idx]

            before_headline = data['BEFORE_HEADLINE']
            before_body    = data['BEFORE_BODY']
            after_headline = data['AFTER_HEADLINE']
            after_body     = data['AFTER_BODY']

            for _ in range(repeat):

                before_body_masking_index = self.get_mask_index(before_body)
                after_body_masking_index = self.get_mask_index(after_body)

                train_data['INDEX'].append(train_step)
                train_data['NEWS1_HEADLINE'].append(before_headline)
                train_data['NEWS1_BODY'].append(before_body)
                train_data['NEWS1_MASKING_INDEX'].append(before_body_masking_index)
                train_data['NEWS2_HEADLINE'].append(after_headline)
                train_data['NEWS2_BODY'].append(after_body)
                train_data['NEWS2_MASKING_INDEX'].append(after_body_masking_index)
                train_data['LABEL'].append(0)

                train_step += 1

            for _ in range(repeat):

                before_body_masking_index = self.get_mask_index(before_body)
                after_body_masking_index = self.get_mask_index(after_body)

                train_data['INDEX'].append(train_step)
                train_data['NEWS2_HEADLINE'].append(before_headline)
                train_data['NEWS2_BODY'].append(before_body)
                train_data['NEWS2_MASKING_INDEX'].append(before_body_masking_index)
                train_data['NEWS1_HEADLINE'].append(after_headline)
                train_data['NEWS1_BODY'].append(after_body)
                train_data['NEWS1_MASKING_INDEX'].append(after_body_masking_index)
                train_data['LABEL'].append(1)

                train_step += 1

        train_df = pd.DataFrame.from_dict(train_data)
        train_df.to_csv(config['train_file'], index=False)

        valid_data = {}
        valid_data['INDEX'] = []
        valid_data['NEWS1_HEADLINE'] = []
        valid_data['NEWS1_BODY'] = []
        valid_data['NEWS2_HEADLINE'] = []
        valid_data['NEWS2_BODY'] = []
        valid_data['LABEL'] = []

        valid_step = 0
        for idx in tqdm.tqdm(self.valid_idx):

            data = self.df.loc[idx]

            before_headline = data['BEFORE_HEADLINE']
            before_body = data['BEFORE_BODY']
            after_headline = data['AFTER_HEADLINE']
            after_body = data['AFTER_BODY']

            valid_data['INDEX'].append(valid_step)
            valid_data['NEWS1_HEADLINE'].append(before_headline)
            valid_data['NEWS1_BODY'].append(before_body)
            valid_data['NEWS2_HEADLINE'].append(after_headline)
            valid_data['NEWS2_BODY'].append(after_body)
            valid_data['LABEL'].append(0)
            
            valid_step += 1

            valid_data['INDEX'].append(valid_step)
            valid_data['NEWS2_HEADLINE'].append(before_headline)
            valid_data['NEWS2_BODY'].append(before_body)
            valid_data['NEWS1_HEADLINE'].append(after_headline)
            valid_data['NEWS1_BODY'].append(after_body)
            valid_data['LABEL'].append(1)

            valid_step += 1

        valid_df = pd.DataFrame.from_dict(valid_data)
        valid_df.to_csv(config['valid_file'], index=False)

    def get_mask_index(self, text):
        tokens = self.tokenizer.tokenize(text)
        length = len(tokens)
        arange = np.arange(length)
        np.random.shuffle(arange)
        idx = arange[:int(length * self.config['mask_ratio'])]
        idx_string = ''
        for i in idx:
            idx_string += f'{i},'
        return idx_string

if __name__ == '__main__':

    config = json.load(open('config.json'))

    SplitData(
            config,
            ratio=0.8,
            repeat=4)
