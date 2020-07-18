import torch
import torchvision
import transformers

import torch.nn as nn

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        bert_config = eval(config['model_config'])
        bert_config = bert_config.from_pretrained(config['tokenizer_path'], output_hidden_states=True)

        self.bert = config
        self.bert = eval(config['model'])
        self.bert = self.bert.from_pretrained(config['tokenizer_path'], config=bert_config)
        if config['freeze'] == 1:
            freeze = False
        elif config['freeze'] == 0:
            freeze = True
        for params in self.bert.parameters():
            params.requires_grad = freeze

        self.linear = nn.Linear(config['n_embed'], 256)
        self.classifier = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.layer = config['layer']

    def forward(self, input_ids,
                segment_ids, attention_ids):

        latent = self.latent(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_ids=attention_ids)
        latent = self.relu(self.linear(latent))
        latent = torch.max(latent, axis=1)[0]
        logit = self.classifier(latent)
        return logit

    def latent(self, input_ids, attention_ids, token_type_ids):
        
        last_hidden, _, hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_ids,
                token_type_ids=token_type_ids)

        latent = hidden[self.layer]
        return latent
