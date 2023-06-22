import os 
import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data 
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_feed import DataFeed
from model import Radar_LSTM
from utils import save_model, load_model, infinite_iter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def build_model(config, device):
    model = Radar_LSTM()
    summary(model, [(config.batch_size, config.x_size, 3)], dtypes=[torch.float])

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.load_model_path:
        model = load_model(model, config.load_model_path)
    model = model.to(device)

    return model, optimizer

def train(config, model, optimizer, train_iter, loss_function, total_steps, device, writer, num_of_train_data):
    model.train()
    model.zero_grad()
    loss_sum = 0.0
    top1_acc = 0.0
    train_summary_steps = config.summary_steps / 5

    for step in range(config.summary_steps):
        radar_infos, label_beam = next(train_iter)

        radar_infos = radar_infos.type(torch.FloatTensor).to(device) 
        label_beam = label_beam.type(torch.LongTensor).to(device)
            
        pred_beam = model(radar_infos)
        loss = loss_function(pred_beam, label_beam)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()

        top1_acc += torch.sum(pred_beam.argmax(1) == label_beam).item()/config.batch_size

        if (step+1) % train_summary_steps == 0:
            loss_sum = loss_sum / train_summary_steps
            top1_acc = top1_acc / train_summary_steps

            # Write training summary
            writer.add_scalar('Loss/train', loss_sum, total_steps + step + 1)
            writer.add_scalar('Top-1_Accuracy/train', top1_acc, total_steps + step + 1)

            print("train [{}] loss: {:.3f}, Top-1 Acc: {:.3f}".format(total_steps + step + 1, loss_sum, top1_acc))
            loss_sum = 0.0
            top1_acc = 0.0

    return model, optimizer

def inference(config, model, data_loader, device, mode='valid', loss_function=None):
    model.eval()
    loss_sum = 0.0
    top1_acc = 0.0
    n = 0
    top5_acc = [0.0] * 5

    y_true = []
    y_pred = []

    per_beam_count = np.zeros((64,)) + 1e-13
    per_beam_true_count = np.zeros((64,))

    for radar_infos, label_beam in data_loader:
        radar_infos = radar_infos.type(torch.FloatTensor).to(device) 
        label_beam = label_beam.type(torch.LongTensor).to(device)

        with torch.no_grad():
                pred_beam = model.forward(radar_infos)
        
        if loss_function:
            loss = loss_function(pred_beam, label_beam)
            loss_sum += loss.item()

        if mode == 'valid':
            top1_acc += torch.sum(pred_beam.argmax(1) == label_beam).item()
        elif mode == 'test':
            y_true.append(label_beam.item())

            y_pred.append(pred_beam.argmax(1).item())

            _, top5s = torch.topk(input=pred_beam, k=5, dim=1, largest=True, sorted=True)
            top5_correct = top5s.eq(label_beam.expand_as(top5s))
            for k in range(1, 6):
                topk_correct = torch.reshape(top5_correct[:, :k], (-1,)).float().sum(0, keepdim=True).item()
                top5_acc[k-1] += (topk_correct == 1.0)

            top1_acc += torch.sum(pred_beam.argmax(1) == label_beam).item()

            per_beam_count[label_beam.item()] += 1
            # Per beam count
            if pred_beam.argmax(1) == label_beam:
                per_beam_true_count[label_beam.item()] += 1

        batch_size = label_beam.size(0)
        n += batch_size

    if mode == 'valid':
        return loss_sum / len(data_loader), top1_acc / n
    elif mode == 'test':
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(64))
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-13)
        return loss_sum / len(data_loader), top1_acc / len(data_loader), [i / n for i in top5_acc], cm, per_beam_count, per_beam_true_count 

def train_process(config):
    setup_seed(0)
    device = torch.device(f'cuda:{config.gpu}')

    # Get output directory ready
    if not os.path.isdir(config.store_model_path):
        os.makedirs(config.store_model_path)

    # Create a summary writer with the specified folder name
    writer = SummaryWriter(os.path.join(config.store_model_path, 'summary'))

    # Prepare training data
    train_feed = DataFeed(config.trn_data_path, config.len_of_sequence, config.x_size, normalize=config.normalize)
    train_loader = data.DataLoader(train_feed, batch_size=config.batch_size, num_workers=8, pin_memory=True)
    train_iter = infinite_iter(train_loader)

    # Prepare validtion data
    val_feed = DataFeed(config.val_data_path, config.len_of_sequence, config.x_size, normalize=config.normalize)
    val_loader = data.DataLoader(val_feed, batch_size=config.batch_size, num_workers=8, pin_memory=True)

    # Build model
    model, optimizer = build_model(config, device)
    print("Finish building model")

    # Define loss function
    loss_function = nn.CrossEntropyLoss()

    total_steps = 0
    num_of_train_data = len(train_feed)
    best_result = 0.0
    
    last_epoch = -1
    print("---training start---")
    while (total_steps < config.num_steps):
        
        num_of_epoch = config.batch_size * total_steps // num_of_train_data 
        if num_of_epoch > 0 and num_of_epoch % 20 == 0 and num_of_epoch != last_epoch:
            last_epoch = num_of_epoch
            config.learning_rate /= 100
            for g in optimizer.param_groups:
                g['lr'] = config.learning_rate
        
        # Train
        train(config, model, optimizer, train_iter, loss_function, total_steps, device, writer, num_of_train_data)

        # Validate
        val_loss, top1_acc = inference(config, model, val_loader, device, loss_function=loss_function)

        total_steps += config.summary_steps

        # Write validation summary
        writer.add_scalar('Loss/validation', val_loss, total_steps)
        writer.add_scalar('Top-1_Accuracy/validation', top1_acc, total_steps)

        print("val [{}] loss: {:.3f}, Top-1 Acc: {:.3f}".format(total_steps, val_loss, top1_acc))

        # Save checkpoint
        num_of_epoch = config.batch_size * total_steps // num_of_train_data
        if num_of_epoch >= 80:
            torch.save(model.state_dict(), f'{config.store_model_path}/Tx_ID_{config.x_size}.ckpt')
            print(f'End of Training')
            break

    writer.close()
    
def test_process(config):
    device = torch.device(f'cuda:{config.gpu}')

    test_feed = DataFeed(config.test_data_path, config.len_of_sequence, config.x_size, normalize=config.normalize)
    test_loader = data.DataLoader(test_feed, batch_size=1, num_workers=8, pin_memory=True)

    # Build model
    model, _ = build_model(config, device)
    print("Finish building model")

    # Test
    _, top1_acc, top5_acc, cm, per_beam_count, per_beam_true_count = inference(config, model, test_loader, device, mode='test')

    ax = sns.heatmap(cm, cmap='YlOrRd')
    ax.set_xlabel('Predicted Beam', fontsize=14)
    ax.set_ylabel('Optimal Beam', fontsize=14)
    ax.set_title('Normalized Confusion Matrix')
    tick_pos = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    tick_labels = [f'{i+1}' for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=12)
    figure = ax.get_figure()
    figure.savefig(os.path.join(os.path.dirname(config.load_model_path), 'cm.png'), dpi=400)

    print("Top-1 Acc: {:.3f}".format(top1_acc))
    print(f"Top-5 Acc: {top5_acc}")
