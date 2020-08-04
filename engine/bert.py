import tqdm
import torch
import torchvision

class TrainingClass:

    def __init__(self, model, optimizer, scheduler, config,
                 trainloader, validloader, writer, device, tokenizer):

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.validloader = validloader
        self.writer = writer
        self.device = device
        self.train_step = 0
        self.scheduler_step = 0
        self.tokenizer = tokenizer

        self.train()

    def scheduler_stepping(self):
        self.scheduler_step += 1
        if self.scheduler_step <= self.config['T_max']:
            self.scheduler[0].step()
        else:
            self.scheduler[1].step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def clean_decode(self, input_ids):
        for input_id in input_ids:
            print(input_id)
            print(self.tokenizer.convert_ids_to_tokens(input_id))
            print('-----')

    def train(self):

        while True:

            self.model.train()
            for input_ids, segment_ids, attention_mask, labels, seq_labels in tqdm.tqdm(self.trainloader):
                self.train_step += 1

                input_ids = input_ids.to(self.device)
                segment_ids = segment_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                seq_labels = seq_labels.to(self.device)

                if self.train_step % self.config['scheduler_step'] == 0:
                    lr = self.scheduler_stepping()
                self.writer.add_scalar('data/lr', self.optimizer.param_groups[0]['lr'], self.train_step)
                self.optimizer.zero_grad()
                mlm_loss, mlm_acc = self.model.mlm_train(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=segment_ids,
                        labels=labels)
                seq_loss, seq_acc = self.model.seq_train(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=segment_ids,
                        labels=seq_labels)
                loss  = mlm_loss * self.config['weight_mlm']
                loss += seq_loss * self.config['weight_seq']
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('data/mlm_loss', mlm_loss.item(), self.train_step)
                self.writer.add_scalar('data/seq_loss', seq_loss.item(), self.train_step)
                self.writer.add_scalar('data/mlm_acc', mlm_acc, self.train_step)
                self.writer.add_scalar('data/seq_acc', seq_acc, self.train_step)

                if self.train_step % self.config['valid_step'] == 0 or self.train_step == self.config['max_iter']:

                    valid_mlm_loss, valid_seq_loss, valid_mlm_acc, valid_seq_acc = self.valid()

                    self.writer.add_scalar('data/valid_mlm_loss', valid_mlm_loss, self.train_step)
                    self.writer.add_scalar('data/valid_seq_loss', valid_seq_loss, self.train_step)
                    self.writer.add_scalar('data/valid_mlm_acc', valid_mlm_acc, self.train_step)
                    self.writer.add_scalar('data/valid_seq_acc', valid_seq_acc, self.train_step)

                    self.save(f'saved/albert_{self.train_step}.pt')

                    if self.train_step == self.config['max_iter']:
                        return None

    def valid(self):

        self.model.eval()
        mlm_loss = 0
        seq_loss = 0
        mlm_acc = 0
        seq_acc = 0

        for input_ids, segment_ids, attention_mask, labels, seq_labels in tqdm.tqdm(self.validloader):

            input_ids = input_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            seq_labels = seq_labels.to(self.device)

            step_mlm_loss, step_mlm_acc = self.model.mlm_train(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=labels)

            step_seq_loss, step_seq_acc = self.model.seq_train(
                    input_ids=labels,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=seq_labels)

            mlm_loss += step_mlm_loss.item()
            seq_loss += step_seq_loss.item()
            mlm_acc  += step_mlm_acc
            seq_acc  += step_seq_acc

        mlm_loss /= len(self.validloader)
        seq_loss /= len(self.validloader)
        mlm_acc  /= len(self.validloader)
        seq_acc  /= len(self.validloader)

        self.model.train()

        return mlm_loss, seq_loss, mlm_acc, seq_acc
