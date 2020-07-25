import tqdm
import torch
import torchvision

class TrainingClass:

    def __init__(self, config):

        self.train_step = 0
        self.config = config

    def valid_combine(self, model, dataloader, device):

        model.eval()
        mlm_loss = 0
        seq_loss = 0
        mlm_acc = 0
        seq_acc = 0

        for input_ids, segment_ids, attention_mask, labels, seq_labels in tqdm.tqdm(dataloader):

            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            seq_labels = seq_labels.to(device)

            step_mlm_loss, step_mlm_acc = model.mlm_train(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=labels)
            step_seq_loss, step_seq_acc = model.seq_train(
                    input_ids=labels,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=seq_labels)

            mlm_loss += step_mlm_loss.item()
            seq_loss += step_seq_loss.item()
            mlm_acc  += step_mlm_acc
            seq_acc  += step_seq_acc

        mlm_loss /= len(dataloader)
        seq_loss /= len(dataloader)
        mlm_acc  /= len(dataloader)
        seq_acc  /= len(dataloader)

        return mlm_loss, seq_loss, mlm_acc, seq_acc

    def train_combine(self, model, dataloader, device, optimizer, writer):

        model.train()
        mlm_loss = 0
        seq_loss = 0
        mlm_acc = 0
        seq_acc = 0

        for input_ids, segment_ids, attention_mask, labels, seq_labels in tqdm.tqdm(dataloader):

            self.train_step += 1
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            seq_labels = seq_labels.to(device)

            optimizer.zero_grad()
            step_mlm_loss, step_mlm_acc = model.mlm_train(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=labels)
            step_seq_loss, step_seq_acc = model.seq_train(
                    input_ids=labels,
                    attention_mask=attention_mask,
                    token_type_ids=segment_ids,
                    labels=seq_labels)
            comb_loss  = step_mlm_loss * self.config['weight_mlm']
            comb_loss += step_seq_loss * self.config['weight_seq']
            comb_loss.backward()
            optimizer.step()

            mlm_loss += step_mlm_loss.item()
            seq_loss += step_seq_loss.item()
            mlm_acc  += step_mlm_acc
            seq_acc  += step_seq_acc

            writer.add_scalar('data/step_mlm_loss', step_mlm_loss.item(), self.train_step)
            writer.add_scalar('data/step_seq_loss', step_seq_loss.item(), self.train_step)
            writer.add_scalar('data/step_mlm_acc', step_mlm_acc, self.train_step)
            writer.add_scalar('data/step_seq_acc', step_seq_acc, self.train_step)

        mlm_loss /= len(dataloader)
        seq_loss /= len(dataloader)
        mlm_acc  /= len(dataloader)
        seq_acc  /= len(dataloader)

        return mlm_loss, seq_loss, mlm_acc, seq_acc


def train_mlm(model, dataloader, device, optimizer):

    loss = 0
    acc = 0
    model.train()
    for input_ids, segment_ids, attention_mask, labels in tqdm.tqdm(dataloader):

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        step_loss, predictions = model.mlm_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=segment_ids,
                labels=labels)
        step_loss.backward()
        optimizer.step()

        loss += step_loss.item()

    loss /= len(dataloader)
    return loss

def load(net, load_path):
    return net.load_state_dict(torch.load(load_path))        

def save(net, save_path):
    return torch.save(net.state_dict(), save_path)
