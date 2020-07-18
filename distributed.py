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

    trainset = dataset.dataset.SequenceDatasetTrain(
            config=config,
            csv_file=config['train_file'])
    trainsampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            num_replicas=hvd.size(),
            rank=hvd.rank())
    trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            sampler=trainsampler)

    validset = dataset.dataset.SequenceDatasetValid(
            config=config,
            csv_file=config['valid_file'])
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

    optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'])
    optimizer = hvd.DistributedOptimizer(
            optimizer=optimizer,
            named_parameters=net.named_parameters())

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=config['gamma'])

    criterion = torch.nn.CrossEntropyLoss()
    hvd.broadcast_parameters(
            net.state_dict(),
            root_rank=0)

    train_step = 0
    valid_step = 0
    for epoch in range(100):

        net.train()
        train_loss, train_acc = 0, 0
        for input_ids, segment_ids, attention_ids, label in tqdm.tqdm(trainloader):

            train_step += 1

            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            attention_ids = attention_ids.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logit = net(
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    attention_ids=attention_ids)
            step_loss = criterion(logit, label)
            step_loss.backward()
            optimizer.step()

            pred = torch.argmax(logit, axis=1)
            pred = pred.eq(label).sum().item() / pred.shape[0]

            writer.add_scalar('data/step_train_loss', step_loss.item(), train_step)
            writer.add_scalar('data/step_train_acc', pred, train_step)
            writer.add_scalar('data/lr', scheduler.get_lr()[0], train_step)

            train_loss += step_loss.item()
            train_acc += pred

            if train_step % config['valid_step'] == 0:
                valid_step += 1
                valid_loss, valid_acc = engine.bert.valid(
                        net=net,
                        criterion=criterion,
                        dataloader=validloader,
                        device=device)
                engine.bert.save(
                        net=net,
                        save_path=f'saved/bert_{train_step}.pt')
                scheduler.step()

                print('-------------------------')
                print(f'valid step : {valid_step}')
                print(f'valid loss : {valid_loss}')
                print(f'valid acc  : {valid_acc}')
                print('-------------------------')

                writer.add_scalar('data/valid_loss', valid_loss, valid_step)
                writer.add_scalar('data/valid_acc', valid_acc, valid_step)

                net.train()

        train_loss /= len(trainloader)
        train_acc /= len(trainloader)

        print('-------------------------')
        print(f'epoch : {epoch}')
        print(f'train loss : {train_loss}')
        print(f'train acc  : {train_acc}')
        print('-------------------------')
        
        writer.add_scalar('data/epoch_train_loss', train_loss, epoch)
        writer.add_scalar('data/epoch_train_acc', train_acc, epoch)

if __name__ == '__main__':
    main()
