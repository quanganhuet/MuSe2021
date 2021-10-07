import time
import os, glob
import numpy as np
import torch
import torch.optim as optim

from eval import evaluate
from loss import get_segment_wise_logits, get_segment_wise_labels


def train(task, model, train_loader, epoch, optimizer, criterion, use_gpu=False):
    start_time = time.time()
    total_loss, total_size = 0, 0
    model.train()
    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)
        if use_gpu:
            model.cuda()
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        preds = model(features, feature_lens)
        # preds = model(vggface,egemaps, feature_lens)

        loss = criterion(preds[:, :, 0], labels[:, :, 0], feature_lens)

        loss.backward()
        optimizer.step()
        
        loss = loss.item() * batch_size
        size = batch_size
        avg_loss = loss / size
        elapsed_time = time.time() - start_time
        print(
            f"Epoch:{epoch:>3} | Batch: {batch:>3} | Lr: {optimizer.state_dict()['param_groups'][0]['lr']:>1.5f}"
            f" | Time used(s): {elapsed_time:>.1f} | Training loss: {avg_loss:>.4f}")

        total_loss += loss
        total_size += size
        start_time = time.time()

    train_loss = total_loss / total_size
    return train_loss


def save_model(model, model_folder, current_seed,val_score):
    for fname in os.listdir(model_folder):
        if fname.startswith(f"model_{current_seed}_CCC"):
            os.remove(os.path.join(model_folder, fname))
    model_file_name = f'model_{current_seed}_CCC_{val_score:>7.4f}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    best_model = torch.load(model_file)

    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, current_seed, use_gpu, criterion,
                regularization=0.0):
    print("Regularization: {}".format(regularization))
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, factor=0.5,
                                                        min_lr=1e-6, verbose=True)
    metric = 'CCC'
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0
    val_losses= []
    train_losses= []
    val_scores=[]
    for epoch in range(1, epochs + 1):
        train_loss = train(task, model, train_loader, epoch, optimizer, criterion, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, criterion, use_gpu)
        print('-' * 50)
        print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} |   [Val] | Loss: {val_loss:>.4f} | [{metric}]: {val_score:>7.4f}')
        print('-' * 50)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        val_scores.append(val_score)
        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            best_model_file = save_model(model, model_path, current_seed, val_score)
        else:
            early_stop += 1
            if early_stop >= 15:
                print(f'Note: target can not be optimized for 15 consecutive epochs, early stop the training process!')
                print('-' * 50)
                break
        lr_scheduler.step(val_loss)

    print(f'Seed {current_seed} | '
          f'Best [Val {metric}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')

    return best_val_loss, best_val_score, best_model_file
