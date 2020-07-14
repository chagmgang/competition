import tqdm
import json
import torch
import transformers
import torchvision

import pandas as pd

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, config):

        self.dataframe = pd.read_csv(csv_file)
        self.tokenizer = eval(config['tokenizer'])
        self.tokenizer = self.tokenizer.from_pretrained(config['tokenizer_path'])

        self.cls_ids = self.tokenizer.convert_tokens_to_ids([config['cls_token']])[0]
        self.pad_ids = self.tokenizer.convert_tokens_to_ids([config['pad_token']])[0]
        self.sep_ids = self.tokenizer.convert_tokens_to_ids([config['sep_token']])[0]

        self.cls_token = config['cls_token']
        self.pad_token = config['pad_token']
        self.sep_token = config['sep_token']

        self.length = 256

    def __getitem__(self, idx):
        data = self.dataframe.loc[idx]

        before_input_ids, before_segment_ids, before_attention_mask = self.combine(
                text1=data['BEFORE_HEADLINE'],
                text2=data['BEFORE_BODY'])

        after_input_ids, after_segment_ids, after_attention_mask = self.combine(
                text1=data['AFTER_HEADLINE'],
                text2=data['AFTER_BODY'])

        before_input_ids = torch.as_tensor(before_input_ids, dtype=torch.long)
        before_segment_ids = torch.as_tensor(before_segment_ids, dtype=torch.long)
        before_attention_mask = torch.as_tensor(before_attention_mask)

        after_input_ids = torch.as_tensor(after_input_ids, dtype=torch.long)
        after_segment_ids = torch.as_tensor(after_segment_ids, dtype=torch.long)
        after_attention_mask = torch.as_tensor(after_attention_mask)

        return before_input_ids, before_segment_ids, before_attention_mask,\
                after_input_ids, after_segment_ids, after_attention_mask

    def combine(self, text1, text2):
        tokenized_text1 = self.tokenizer.tokenize(text1)
        tokenized_text2 = self.tokenizer.tokenize(text2)

        input_ids = [self.cls_token]
        segment_ids = [0]
        attention_mask = [1]

        input_ids.extend(tokenized_text1)
        while len(segment_ids) < len(input_ids):
            segment_ids.append(0)
            attention_mask.append(1)

        input_ids.append(self.sep_token)
        segment_ids.append(0)
        attention_mask.append(1)

        input_ids.extend(tokenized_text2)
        while len(segment_ids) < len(input_ids):
            segment_ids.append(1)
            attention_mask.append(1)

        input_ids.append(self.sep_token)
        segment_ids.append(1)
        attention_mask.append(1)

        input_ids = input_ids[:self.length]
        segment_ids = segment_ids[:self.length]
        attention_mask = attention_mask[:self.length]

        while len(input_ids) < self.length:
            input_ids.append(self.pad_token)
            segment_ids.append(0)
            attention_mask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)

        return input_ids, segment_ids, attention_mask

    def __len__(self):
        return len(self.dataframe)

if __name__ == '__main__':

    config = json.load(open('config.json'))

    dataset = SequenceDataset(
            csv_file='training.csv',
            config=config)

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=3,
            num_workers=8)

    before_input_ids, before_segment_ids, before_attention_mask,\
            after_input_ids, after_segment_ids, after_attention_mask = next(iter(dataloader))

    model = eval(config['model'])
    model = model.from_pretrained(config['tokenizer_path'])

    latent = model(
            input_ids=before_input_ids,
            token_type_ids=before_segment_ids,
            attention_mask=before_attention_mask)
    latent = latent[0]
    print(latent.shape)
