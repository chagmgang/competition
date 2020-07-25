import os
import json
import tqdm
import torch
import argparse
import torchvision
import transformers

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
            name=f'albert_mlm_{hvd.local_rank()}')
    writer.add_hparams(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = dataset.dataset.MaskedLMDataset(
            csv_file=config['mask_train_file'],
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

    validset = dataset.dataset.MaskedLMDataset(
            csv_file=config['mask_valid_file'],
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

    net = model.bert.Model(
            config=config).to(device)

    optimizer = transformers.AdamW(
            params=net.parameters(),
            lr=config['mlm_lr'],
            weight_decay=config['weight_decay'])
    optimizer = hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=net.named_parameters())

    hvd.broadcast_parameters(
            net.state_dict(),
            root_rank=0)

    training_method = engine.bert.TrainingClass(
            config=config)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=config['gamma'])

    train_step = 0
    for epoch in range(100):

        train_mlm_loss, train_seq_loss, train_mlm_acc, train_seq_acc = \
                training_method.train_combine(
                        model=net,
                        dataloader=trainloader,
                        device=device,
                        optimizer=optimizer,
                        writer=writer)

        valid_mlm_loss, valid_seq_loss, valid_mlm_acc, valid_seq_acc = \
                training_method.valid_combine(
                        model=net,
                        dataloader=validloader,
                        device=device)

        print('-------------')
        print(f'epoch : {epoch}')
        print(f'train mlm loss  : {train_mlm_loss}')
        print(f'train mlm acc   : {train_mlm_acc}')
        print(f'train seq loss  : {train_seq_loss}')
        print(f'train seq acc   : {train_seq_acc}')
        print(f'valid mlm loss  : {valid_mlm_loss}')
        print(f'valid mlm acc   : {valid_mlm_acc}')
        print(f'valid seq loss  : {valid_seq_loss}')
        print(f'valid seq acc   : {valid_seq_acc}')
        print(f'lr              : {scheduler.get_lr()[0]}')
        print('-------------')

        writer.add_scalar('data/train_mlm_loss', train_mlm_loss, epoch)
        writer.add_scalar('data/train_seq_loss', train_seq_loss, epoch)
        writer.add_scalar('data/train_mlm_acc' , train_mlm_acc, epoch)
        writer.add_scalar('data/train_seq_acc' , train_seq_acc, epoch)
        writer.add_scalar('data/valid_mlm_loss', valid_mlm_loss, epoch)
        writer.add_scalar('data/valid_seq_loss', valid_seq_loss, epoch)
        writer.add_scalar('data/valid_mlm_acc' , valid_mlm_acc, epoch)
        writer.add_scalar('data/valid_seq_acc' , valid_seq_acc, epoch)
        writer.add_scalar('data/lr'            , scheduler.get_lr()[0], epoch)

        scheduler.step()

        engine.bert.save(
                net=net,
                save_path=f'saved/albert_mlm_{epoch+1}.pt')

if __name__ == '__main__':
    main()
