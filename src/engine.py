# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:40:24 2020

@author: Gautam_Pai
"""


import tqdm
import torch
import config


def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm.tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm.tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            batch_preds, loss = model(**data)
            fin_preds.append(batch_preds)
            fin_loss += loss.item()
        return fin_preds, fin_loss / len(data_loader)
