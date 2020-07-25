import os
import json
import tqdm
import torch
import argparse
import torchvision

import engine.bert
import model.bert
import dataset.dataset

import horovod.torch as hvd
import tensorboard.summary as tb_summary
import tensorflow as tf
import numpy as np

class Logging:

    def __init__(self, user, name):

        assert not os.path.exists(f'{user}/{name}')

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(f'{user}/{name}')

    def add_scalar(self, field, data, global_step):
        write = tb_summary.scalar_pb(field, data)
        self.writer.add_summary(write, global_step=global_step)

    def add_hparams(self, data):
        data = [[key, json.dumps(data[key])] for key in data.keys()]
        write = tb_summary.text_pb('hparams', data)
        self.writer.add_summary(write, global_step=0)

def main():

    hvd.init()
    config = json.load(open('config.json'))
    torch.cuda.set_device(hvd.local_rank())

    writer = Logging(
            user='ckg',
            name=f'albert_{hvd.local_rank()}')
    writer.add_hparams(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = dataset.dataset.TotalDataset(
        csv_file=config['train_file'],
        config=config)
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=hvd.size(),
        rank=hvd.rank())
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sampler=trainsampler)

    validset = dataset.dataset.TotalDataset(
        csv_file=config['valid_file'],
        config=config)
    validsampler = torch.utils.data.distributed.DistributedSampler(
        validset,
        num_replicas=hvd.size(),
        rank=hvd.rank())
    validloader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sampler=validsampler)

    net = model.bert.Model(config=config).to(device)

    optimizer = torch.optim.Adam(
        params=net.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'])
    optimizer = hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=net.named_parameters())

    hvd.broadcast_parameters(
            net.state_dict(),
            root_rank=0)

    train_step = 0
    for epoch in range(100):

        train_mlm_loss, train_seq_loss, train_seq_acc = 0, 0, 0
        net.train()
        for data in tqdm.tqdm(trainloader):

            train_step += 1

            forward_input_ids = data[0].to(device)
            forward_segment   = data[1].to(device)
            forward_attention = data[2].to(device)
            forward_label     = data[3].to(device)

            reverse_input_ids = data[4].to(device)
            reverse_segment   = data[5].to(device)
            reverse_attention = data[6].to(device)
            reverse_label     = data[7].to(device)

            optimizer.zero_grad()
            forward_mlm_loss, forward_seq_loss, forward_seq_logit = net(
                input_ids=forward_input_ids,
                attention_mask=forward_attention,
                token_type_ids=forward_segment,
                mask_label=forward_input_ids,
                sequence_label=forward_label)
            reverse_mlm_loss, reverse_seq_loss, reverse_seq_logit = net(
                input_ids=reverse_input_ids,
                attention_mask=reverse_attention,
                token_type_ids=reverse_segment,
                mask_label=reverse_input_ids,
                sequence_label=reverse_label)
            
            loss  = forward_mlm_loss * 1.0
            loss += forward_seq_loss * 1.0
            loss += reverse_mlm_loss * 1.0
            loss += reverse_seq_loss * 1.0
            loss.backward()
            optimizer.step()

            forward_pred = torch.argmax(forward_seq_logit, axis=1)
            forward_pred = forward_pred.eq(forward_label).sum().item() / forward_pred.shape[0]
            reverse_pred = torch.argmax(reverse_seq_logit, axis=1)
            reverse_pred = reverse_pred.eq(reverse_label).sum().item() / reverse_pred.shape[0]

            train_mlm_loss += (forward_mlm_loss + reverse_mlm_loss).item()
            train_seq_loss += (forward_seq_loss + reverse_seq_loss).item()
            train_seq_acc  += (forward_pred + reverse_pred) / 2

            writer.add_scalar(
                'data/step_mlm_loss',
                (forward_mlm_loss + reverse_mlm_loss).item(),
                train_step)
            writer.add_scalar(
                'data/step_seq_loss',
                (forward_seq_loss + reverse_seq_loss).item(),
                train_step)
            writer.add_scalar(
                'data/step_seq_acc',
                (forward_pred + reverse_pred) / 2,
                train_step)

            if train_step % config['valid_step'] == 0:
                if hvd.local_rank() == 0:
                    torch.save(net.state_dict(), f'saved/albert_{train_step}.pt') 

        train_mlm_loss /= len(trainloader)
        train_seq_loss /= len(trainloader)
        train_seq_acc  /= len(trainloader)

        writer.add_scalar('data/epoch_mlm_loss', train_mlm_loss, epoch)
        writer.add_scalar('data/epoch_seq_loss', train_seq_loss, epoch)
        writer.add_scalar('data/epoch_seq_acc',  train_seq_acc,  epoch)

        valid_mlm_loss, valid_seq_loss, valid_seq_acc = 0, 0, 0
        net.eval()
        for data in tqdm.tqdm(validloader):

            forward_input_ids = data[0].to(device)
            forward_segment   = data[1].to(device)
            forward_attention = data[2].to(device)
            forward_label     = data[3].to(device)

            reverse_input_ids = data[4].to(device)
            reverse_segment   = data[5].to(device)
            reverse_attention = data[6].to(device)
            reverse_label     = data[7].to(device)

            forward_input_ids = data[0].to(device)
            forward_segment   = data[1].to(device)
            forward_attention = data[2].to(device)
            forward_label     = data[3].to(device)

            reverse_input_ids = data[4].to(device)
            reverse_segment   = data[5].to(device)
            reverse_attention = data[6].to(device)
            reverse_label     = data[7].to(device)

            forward_mlm_loss, forward_seq_loss, forward_seq_logit = net(
                input_ids=forward_input_ids,
                attention_mask=forward_attention,
                token_type_ids=forward_segment,
                mask_label=forward_input_ids,
                sequence_label=forward_label)
            reverse_mlm_loss, reverse_seq_loss, reverse_seq_logit = net(
                input_ids=reverse_input_ids,
                attention_mask=reverse_attention,
                token_type_ids=reverse_segment,
                mask_label=reverse_input_ids,
                sequence_label=reverse_label)

            loss  = forward_mlm_loss * 1.0
            loss += forward_seq_loss * 1.0
            loss += reverse_mlm_loss * 1.0
            loss += reverse_seq_loss * 1.0

            valid_mlm_loss += (forward_mlm_loss + reverse_mlm_loss).item()
            valid_seq_loss += (forward_seq_loss + reverse_seq_loss).item()
            valid_seq_acc  += (forward_pred + reverse_pred) / 2

        valid_mlm_loss /= len(validloader)
        valid_seq_loss /= len(validloader)
        valid_seq_acc  /= len(validloader)

        writer.add_scalar('data/valid_mlm_loss', valid_mlm_loss, epoch)
        writer.add_scalar('data/valid_seq_loss', valid_seq_loss, epoch)
        writer.add_scalar('data/valid_seq_acc',  valid_seq_acc,  epoch)

        print('################')
        print(f'epoch             : {epoch}')
        print(f'train_mlm_loss    : {train_mlm_loss}')
        print(f'train_seq_loss    : {train_seq_loss}')
        print(f'train_seq_acc     : {train_seq_acc}')
        print(f'valid_mlm_loss    : {valid_mlm_loss}')
        print(f'valid_seq_loss    : {valid_seq_loss}')
        print(f'valid_seq_acc     : {valid_seq_acc}')
        print('################')

if __name__ == '__main__':
    main()
