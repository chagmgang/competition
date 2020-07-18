import tqdm
import torch
import torchvision

def save(net, save_path):
    return torch.save(net.state_dict(), save_path)

def valid(net, criterion, dataloader, device):

    loss = 0
    acc = 0
    net.eval()

    for input_ids, segment_ids, attention_ids, label in tqdm.tqdm(dataloader):

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attention_ids = attention_ids.to(device)
        label = label.to(device)

        logit = net(
                input_ids=input_ids,
                segment_ids=segment_ids,
                attention_ids=attention_ids)

        step_loss = criterion(logit, label)
        pred = torch.argmax(logit, axis=1)
        pred = pred.eq(label).sum().item() / pred.shape[0]

        loss += step_loss.item()
        acc += pred

    loss /= len(dataloader)
    acc /= len(dataloader)
    net.train()
    return loss, acc


def train(net, optimizer, criterion, dataloader, device):

    loss = 0
    acc = 0
    net.train()
    for before_input, before_segment, before_attention, after_input, after_segment, after_attention, label in tqdm.tqdm(dataloader):

        before_input = before_input.to(device)
        before_segment = before_segment.to(device)
        before_attention = before_attention.to(device)
        after_input = after_input.to(device)
        after_segment = after_segment.to(device)
        after_attention = after_attention.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logit = net(
                before_input=before_input,
                before_segment=before_segment,
                before_attention=before_attention,
                after_input=after_input,
                after_segment=after_segment,
                after_attention=after_attention)
        step_loss = criterion(logit, label)
        step_loss.backward()
        optimizer.step()

        pred = torch.argmax(logit, axis=1)
        pred = pred.eq(label).sum().item() / pred.shape[0]

        loss += step_loss.item()
        acc += pred

    loss /= len(dataloader)
    acc /= len(dataloader)

    return loss, acc
