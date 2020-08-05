# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 00:21:37 2020

@author: Gautam_Pai
"""


import os
import glob
import torch
import numpy as np

from sklearn import preprocessing, model_selection, metrics

import config
import dataset
from model import CaptchaModel
import engine
import pprint


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))

    targets_orig = [x.split("\\")[-1][:-4] for x in image_files]

    targets = [[y for y in x] for x in targets_orig]

    targets_flat = [c for clist in targets for c in clist]

    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(targets_flat)
    targets_enc = [label_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1
    # print(targets_enc)
    # print(label_enc.classes_)

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        train_orig_targets,
        test_orig_targets,
    ) = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    model = CaptchaModel(num_chars=len(label_enc.classes_)).cuda()
    model.to(torch.device(config.DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp, label_enc)
            valid_cap_preds.extend(current_preds)
        pprint.pprint(list(zip(test_orig_targets, valid_cap_preds))[:10])
        test_dup_rem = [remove_duplicates(c) for c in test_orig_targets]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_cap_preds)
        print(
            f"EPOCH: {epoch}.train_loss:{train_loss},valid_loss:{valid_loss}, Accuracy={accuracy}"
        )
        scheduler.step(valid_loss)

    #torch.save(model.state_dict, "captcha_model.pt")


if __name__ == "__main__":
    run_training()
