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
            lr=config['start_lr'],
            weight_decay=config['weight_decay'])
    optimizer = hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=net.named_parameters())

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config['T_max'],
            eta_min=config['eta_min'])
    scheduler_2 = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=config['base_lr'],
            max_lr=config['max_lr'],
            step_size_up=config['step_size_up'],
            step_size_down=config['step_size_down'],
            cycle_momentum=False)

    scheduler = [scheduler_1, scheduler_2]

    hvd.broadcast_parameters(
            net.state_dict(),
            root_rank=0)

    training_method = engine.bert.TrainingClass(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            trainloader=trainloader,
            validloader=validloader,
            writer=writer,
            device=device,
            tokenizer=trainset.tokenizer)

if __name__ == '__main__':
    main()
