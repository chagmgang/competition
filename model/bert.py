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
        self.classifier = nn.Linear(config['n_embed'] * 2, 2)

        self.layer = config['layer']

    def forward(self, before_input, before_segment, before_attention,
                after_input, after_segment, after_attention):

        before_latent = self.latent(
                input_ids=before_input,
                attention_mask=before_attention,
                token_type_ids=before_segment)

        after_latent = self.latent(
                input_ids=after_input,
                attention_mask=after_attention,
                token_type_ids=after_segment)

        before_latent = before_latent[:, 0]
        after_latent = after_latent[:, 0]

        cat = torch.cat([before_latent, after_latent], axis=1)
        logit = self.classifier(cat)
        return logit

    def latent(self, input_ids, attention_mask, token_type_ids):
        
        last_hidden, _, hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

        latent = hidden[self.layer]
        return latent
