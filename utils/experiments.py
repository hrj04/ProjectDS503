import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torchmetrics import MeanAbsolutePercentageError
import matplotlib.pyplot as plt


def visualize_prediction(true_y, pred_y):
    plt.figure()
    plt.plot(true_y, label='True Data', marker="o")
    plt.plot(pred_y, label='Predictions', linestyle='--', color='r',  marker="o")
    plt.legend()
    plt.xlabel("Time Step")
    plt.show()


def print_score(score, base_mae, base_rmse):
    mae_mean, rmse_mean = np.array(score).mean(0)
    mae_std, rmse_std = np.array(score).std(0)
    
    change_mae = (mae_mean - base_mae)/base_mae * 100
    change_rmse = (rmse_mean - base_rmse)/base_rmse * 100
    
    score_df = pd.DataFrame(score, columns=["MAE", "MSE"])
    print()
    print(score_df)
    print("----------------------------------------------------")
    print(f"baseline mae : {base_mae:0.4f}, baseline rmse : {base_rmse:0.4f}")
    print(f"MAE : {mae_mean:0.4f}({mae_std:0.4f})({change_mae:0.4f}%) \nMSE : {rmse_mean:0.4f}({rmse_std:0.4f})({change_rmse:0.4f}%)")
    print("----------------------------------------------------")
    return mae_mean, mae_std, change_mae, rmse_mean, rmse_std, change_rmse


def train_model(model, dataloader, criterion, optimizer, num_epochs, description, device):
    model.train()
    with tqdm(range(num_epochs), total=num_epochs) as pbar:
        for _ in pbar:
            for data, *_ in dataloader:
                x = data[:,:-1,:].float().to(device)
                y = data[:,-1:,0].float().to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            pbar.set_description(f"{description} loss: {loss.item():.6f}")


def evaluate_model_nonstationary(model, dataloader, device, desc=""):
    model.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data, mean, std in dataloader:
            data = data.to(device)
            mean = mean[:,:,:1].to(device)
            std = std[:,:,:1].to(device)
            
            x = data[:, :-1, :].float()
            y_true = data[:, -1:, :1].float()
            y_pred = model(x).view(-1,1,1)
            
            y_true = torch.clamp(y_true, -0.999999, 0.999999)
            y_pred = torch.clamp(y_pred, -0.999999, 0.999999)
            y_true = torch.arctanh(y_true)
            y_pred = torch.arctanh(y_pred)
            
            y_true_unnorm = y_true * (std + 1e-8) + mean
            y_pred_unnorm = y_pred * (std + 1e-8) + mean
            predictions.append(y_pred_unnorm.cpu().detach())
            ground_truth.append(y_true_unnorm.cpu().detach())

    predictions = torch.concatenate(predictions).squeeze()
    ground_truth = torch.concatenate(ground_truth).squeeze()
    mae_loss = nn.L1Loss()(predictions, ground_truth).item()
    rmse_loss = np.sqrt(nn.MSELoss()(predictions, ground_truth)).item()
    
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth


def evaluate_model_stationary(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data_st, data, mean, std in dataloader:
            data_st = data_st.to(device)
            data = data.to(device)
            mean = mean[:,:,:1].to(device)
            std = std[:,:,:1].to(device)
            
            x_diff = data_st[:, :-1, :].float()
            y_true_diff = data_st[:, -1:, :1].float()
            y_pred_diff = model(x_diff).view(-1,1,1)
            
            y_true_diff = torch.clamp(y_true_diff, -0.999999, 0.999999)
            y_pred_diff = torch.clamp(y_pred_diff, -0.999999, 0.999999)
            y_true_diff = torch.arctanh(y_true_diff)
            y_pred_diff = torch.arctanh(y_pred_diff)

            y_true_diff_unnorm = y_true_diff * (std + 1e-8) + mean
            y_pred_diff_unnorm = y_pred_diff * (std + 1e-8) + mean

            y_true = data[:,-2:-1,:1] + y_true_diff_unnorm
            y_pred = data[:,-2:-1,:1] + y_pred_diff_unnorm

            predictions.append(y_pred.cpu().detach())
            ground_truth.append(y_true.cpu().detach())

    predictions = torch.concatenate(predictions).squeeze()
    ground_truth = torch.concatenate(ground_truth).squeeze()
    mae_loss = l1loss(predictions, ground_truth).item()
    rmse_loss = np.sqrt(l2loss(predictions, ground_truth).item())
    
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth


















def evaluate_norm(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data, mean, std in dataloader:
            data = data.to(device)
            mean = mean[:,:,:1].to(device)
            std = std[:,:,:1].to(device)
            batch_size = len(data)
            
            x = data[:, :-1, :].float()
            y_true = data[:, -1:, :1].float()
            y_pred = model(x).view(-1,1,1)
            
            y_true = torch.clamp(y_true, -0.999999, 0.999999)
            y_pred = torch.clamp(y_pred, -0.999999, 0.999999)
            y_true = torch.arctanh(y_true)
            y_pred = torch.arctanh(y_pred)
            
            y_true_unnorm = y_true * (std + 1e-8) + mean
            y_pred_unnorm = y_pred * (std + 1e-8) + mean

            total_l1 += l1loss(y_pred_unnorm, y_true_unnorm) * batch_size
            total_l2 += l2loss(y_pred_unnorm, y_true_unnorm) * batch_size

            predictions.append(y_pred_unnorm.cpu().numpy())
            ground_truth.append(y_true_unnorm.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth



def evaluate_stock_norm(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data, mean, std in dataloader:
            data = data.to(device)
            mean = mean[:,:,:1].to(device)
            std = std[:,:,:1].to(device)
            batch_size = len(data)
            
            x = data[:, :-1, :].float()
            y_true = data[:, -1:, :1].float()
            y_pred = model(x).view(-1,1,1)
            
            y_true = torch.clamp(y_true, -0.999999, 0.999999)
            y_pred = torch.clamp(y_pred, -0.999999, 0.999999)
            y_true = torch.arctanh(y_true)
            y_pred = torch.arctanh(y_pred)
            
            y_true_unnorm = y_true * (std + 1e-8) + mean
            y_pred_unnorm = y_pred * (std + 1e-8) + mean

            total_l1 += l1loss(y_pred_unnorm, y_true_unnorm) * batch_size
            total_l2 += l2loss(y_pred_unnorm, y_true_unnorm) * batch_size

            predictions.append(y_pred_unnorm.cpu().numpy())
            ground_truth.append(y_true_unnorm.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth


def evaluate_stock_norm_diff(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data_diff, data, mean, std in dataloader:
            data_diff = data_diff.to(device)
            data = data.to(device)
            mean = mean[:,:,:1].to(device)
            std = std[:,:,:1].to(device)
            batch_size = len(data)
            
            x_diff = data_diff[:, :-1, :].float()
            y_true_diff = data_diff[:, -1:, :1].float()
            y_pred_diff = model(x_diff).view(-1,1,1)
            
            y_true_diff = torch.clamp(y_true_diff, -0.999999, 0.999999)
            y_pred_diff = torch.clamp(y_pred_diff, -0.999999, 0.999999)
            y_true_diff = torch.arctanh(y_true_diff)
            y_pred_diff = torch.arctanh(y_pred_diff)

            y_true_diff_unnorm = y_true_diff * (std + 1e-8) + mean
            y_pred_diff_unnorm = y_pred_diff * (std + 1e-8) + mean

            y_true = data[:,-2:-1,:1] + y_true_diff_unnorm
            y_pred = data[:,-2:-1,:1] + y_pred_diff_unnorm
            
            total_l1 += l1loss(y_pred, y_true) * batch_size
            total_l2 += l2loss(y_pred, y_true) * batch_size

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y_true.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth



def evaluate_energy_norm(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data, min, max in dataloader:
            data = data.to(device)
            min = min[:,:,:1].to(device)
            max = max[:,:,:1].to(device)
            batch_size = len(data)
            
            x = data[:, :-1, :].float()
            y_true = data[:, -1:, :1].float()
            y_pred = model(x).view(-1,1,1)
            
            y_true = torch.clamp(y_true, -0.999999, 0.999999)
            y_pred = torch.clamp(y_pred, -0.999999, 0.999999)
            y_true = torch.arctanh(y_true)
            y_pred = torch.arctanh(y_pred)
            
            y_true_unnorm = y_true * (max-min + 1e-8) + min
            y_pred_unnorm = y_pred * (max-min + 1e-8) + min

            total_l1 += l1loss(y_pred_unnorm, y_true_unnorm) * batch_size
            total_l2 += l2loss(y_pred_unnorm, y_true_unnorm) * batch_size

            predictions.append(y_pred_unnorm.cpu().numpy())
            ground_truth.append(y_true_unnorm.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth


def evaluate_energy_norm_diff(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data_diff, data, min, max in dataloader:
            data_diff = data_diff.to(device)
            data = data.to(device)
            min = min[:,:,:1].to(device)
            max = max[:,:,:1].to(device)
            batch_size = len(data)
            
            x_diff = data_diff[:, :-1, :].float()
            y_true_diff = data_diff[:, -1:, :1].float()
            y_pred_diff = model(x_diff).view(-1,1,1)
            
            y_true_diff = torch.clamp(y_true_diff, -0.999999, 0.999999)
            y_pred_diff = torch.clamp(y_pred_diff, -0.999999, 0.999999)
            y_true_diff = torch.arctanh(y_true_diff)
            y_pred_diff = torch.arctanh(y_pred_diff)

            y_true_diff_unnorm = y_true_diff * (max-min) + min
            y_pred_diff_unnorm = y_pred_diff * (max-min) + min

            y_true = data[:,-2:-1,:1] + y_true_diff_unnorm
            y_pred = data[:,-2:-1,:1] + y_pred_diff_unnorm
            
            total_l1 += l1loss(y_pred, y_true) * batch_size
            total_l2 += l2loss(y_pred, y_true) * batch_size

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y_true.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f}")

    return mae_loss, rmse_loss, predictions, ground_truth



def evaluate_ETTh(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data, min, max in dataloader:
            data = data.to(device)
            min = min[:,:,:1].to(device)
            max = max[:,:,:1].to(device)
            batch_size = len(data)
            
            x = data[:, :-1, :].float()
            y_true = data[:, -1:, :1].float()
            y_pred = model(x).view(-1,1,1)
            
            y_true_unnorm = y_true *(max-min) + min
            y_pred_unnorm = y_pred *(max-min) + min
            
            total_l1 += l1loss(y_pred_unnorm, y_true_unnorm) * batch_size
            total_l2 += l2loss(y_pred_unnorm, y_true_unnorm) * batch_size

            predictions.append(y_pred_unnorm.cpu().numpy())
            ground_truth.append(y_true_unnorm.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f} \t")

    return mae_loss, rmse_loss, predictions, ground_truth


def evaluate_ETTh_stationary(model, dataloader, device, desc=""):
    model.eval()
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    
    total_l1 = 0
    total_l2 = 0
    predictions, ground_truth = [], []
    with torch.no_grad():
        for data_diff, data, min, max in dataloader:
            data_diff = data_diff.to(device)
            data = data.to(device)
            min = min[:,:,:1].to(device)
            max = max[:,:,:1].to(device)
            batch_size = len(data)
            
            x_diff = data_diff[:, :-1, :].float()
            y_true_diff = data_diff[:, -1:, :1].float()
            y_pred_diff = model(x_diff).view(-1,1,1)
            
            y_true_diff_unnorm = y_true_diff *(max-min) + min
            y_pred_diff_unnorm = y_pred_diff *(max-min) + min
            
            y_true = data[:,-2:-1,:1] + y_true_diff_unnorm
            y_pred = data[:,-2:-1,:1] + y_pred_diff_unnorm
            

            total_l1 += l1loss(y_pred, y_true) * batch_size
            total_l2 += l2loss(y_pred, y_true) * batch_size

            predictions.append(y_pred.cpu().numpy())
            ground_truth.append(y_true.cpu().numpy())

    n_data = len(dataloader.dataset)
    total_l1 /= n_data
    total_l2 /= n_data
    
    predictions = np.concatenate(predictions).squeeze()
    ground_truth = np.concatenate(ground_truth).squeeze()
    
    mae_loss = total_l1.item()
    rmse_loss = np.sqrt(total_l2.item())
    print(f"{desc} : MAE loss: {mae_loss:0.4f} \t RMSE Loss : {rmse_loss:0.4f} \t")

    return mae_loss, rmse_loss, predictions, ground_truth

