import torch
import torchvision
import transformers

import torch.nn as nn

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.bert = config
        self.bert = eval(config['model'])
        self.bert = self.bert.from_pretrained(config['tokenizer_path'])

