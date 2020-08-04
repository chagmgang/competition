import torch
import torchvision
import transformers

import torch.nn as nn

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.albert_mlm = eval(config['model'])
        self.albert_mlm = self.albert_mlm.from_pretrained(
            config['tokenizer_path'])

        self.classifier1 = nn.Linear(self.config['n_embed'], self.config['n_embed'])
        self.classifier2 = nn.Linear(self.config['n_embed'], 2)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()

    def mlm_train(self, input_ids, attention_mask,
                  token_type_ids, labels):

        mlm_loss, predictions, hiddens = self.albert_mlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                output_hidden_states=True)

        mlm_acc = self.mlm_criterion(
                input_ids=input_ids,
                labels=labels,
                predictions=predictions)

        return mlm_loss, mlm_acc

    def seq_train(self, input_ids, attention_mask,
                  token_type_ids, labels):
        _, hiddens = self.albert_mlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True)
        hidden = hiddens[self.config['layer']]
        seq_logit = self.prediction_seq(hidden)
        seq_loss, seq_acc = self.seq_criterion(seq_logit, labels)
        return seq_loss, seq_acc

    def mlm_criterion(self, input_ids, predictions, labels):

        
        predictions = torch.argmax(predictions, axis=2)
        predictions = predictions.view(-1)
        labels = labels.view(-1)

        acc = predictions.eq(labels)
        acc = acc.sum().item() / acc.shape[0]
        return acc

    def forward(self, input_ids, attention_mask,
                 token_type_ids, labels, seq_labels):

        mlm_loss, predictions, hiddens = self.albert_mlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                output_hidden_states=True)

        mlm_acc = self.mlm_criterion(
                input_ids=input_ids,
                labels=labels,
                predictions=predictions)

        seq_logit = self.prediction_seq(hiddens[self.config['layer']])
        seq_loss, seq_acc = self.seq_criterion(seq_logit, seq_labels)

        return mlm_loss, mlm_acc, seq_loss, seq_acc

    def prediction(self, input_ids, attention_mask, token_type_ids):

        predictions, hiddens = self.albert_mlm(
	        input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True)
        
        seq_logit = self.prediction_seq(hiddens[self.config['layer']])
        return seq_logit

    def seq_criterion(self, seq_logits, seq_labels):

        loss = self.criterion(seq_logits, seq_labels)
        pred = torch.argmax(seq_logits, axis=1)
        pred = pred.eq(seq_labels).sum().item() / pred.shape[0]

        return loss, pred

    def prediction_seq(self, hidden):
        
        hidden = hidden[:, 0]
        hidden = self.classifier1(hidden)
        hidden = self.relu(hidden)
        hidden = self.classifier2(hidden)

        return hidden
