import os
import torch
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import Dataset
        

class TS_Dataset(Dataset):
    def __init__(self,
                 window,
                 train_test,
                 train_ratio,
                 ):
        super().__init__()
        self.window = window
        self.train_test = train_test
        self.train_ratio = train_ratio
        
    def _train_test_selection(self, raw_data, train_test):
        train_number = int(self.train_ratio*len(raw_data))
        if train_test == "train":
            raw_data = raw_data[:train_number, :]
        else:
            raw_data = raw_data[train_number:, :]
            
        return raw_data
   
    def _generate_ts_seq_data(self, raw_data, window):
        raise NotImplementedError()
         
    def _extract_sliding_windows(self, raw_data, window):
        sample_n = len(raw_data) - window + 1
        n_feature = raw_data.shape[-1]
        data = torch.zeros(sample_n, window, n_feature)
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
            
        return data

    def _differencing(self, raw_data, n_order, axis):
        data_st = torch.from_numpy(np.diff(raw_data, n_order, axis))
        
        return data_st

    def _normalize(self, data):
        epsilon = 1e-8
        past_data= data[:,:-1,:]
        mean = past_data.mean(axis=1, keepdim=True)
        std = past_data.std(axis=1, keepdim=True)
        normalized_data = (data-mean)/(std+epsilon)
        
        return normalized_data, mean, std
   
    def _save_ground_truth(self):
        raise NotImplementedError()

class StationaryDataset(TS_Dataset):
    def __init__(self, 
                 window, 
                 train_test, 
                 train_ratio,
                 ):
        super().__init__(window, train_test, train_ratio)
        
    def _generate_ts_seq_data(self, raw_data, window):
        data = self._extract_sliding_windows(raw_data, window)
        data_st = self._differencing(data, n_order=1, axis=1)
        data_st_norm, mean, std = self._normalize(data_st)
        data_st_norm = torch.tanh(data_st_norm) # addressing value exploding

        return data_st_norm, data, mean, std

    def _save_ground_truth(self):
        np.save(os.path.join(self.dir, f"raw_data_{self.window}_{self.train_test}.npy"), self.raw_data)
        np.save(os.path.join(self.dir, f"data_st_norm_{self.window}_{self.train_test}.npy"), self.data_st_norm)
        np.save(os.path.join(self.dir, f"data_{self.window}_{self.train_test}.npy"), self.data)
        np.save(os.path.join(self.dir, f"mean_{self.window}_{self.train_test}.npy"), self.mean)
        np.save(os.path.join(self.dir, f"std_{self.window}_{self.train_test}.npy"), self.std)
    
    def _load_ground_truth(self, window, train_test):
        raw_data = torch.from_numpy(np.load(os.path.join(self.dir, f"raw_data_{window}_{train_test}.npy")))
        data_st_norm = torch.from_numpy(np.load(os.path.join(self.dir, f"data_st_norm_{window}_{train_test}.npy")))
        data = torch.from_numpy(np.load(os.path.join(self.dir, f"data_{window}_{train_test}.npy")))
        mean = torch.from_numpy(np.load(os.path.join(self.dir, f"mean_{window}_{train_test}.npy")))
        std = torch.from_numpy(np.load(os.path.join(self.dir, f"std_{window}_{train_test}.npy")))

        return raw_data, data_st_norm, data, mean, std

class NonStationaryDataset(TS_Dataset):
    def __init__(self, 
                 window, 
                 train_test, 
                 train_ratio,
                 ):
        super().__init__(window, train_test, train_ratio)

    def _generate_ts_seq_data(self, raw_data, window):
        data = self._extract_sliding_windows(raw_data, window)
        data_norm, mean, std = self._normalize(data)
        data_norm = torch.tanh(data_norm) # addressing value exploding

        return data_norm, mean, std

    def _save_ground_truth(self):
        np.save(os.path.join(self.dir, f"raw_data_{self.window}_{self.train_test}.npy"), self.raw_data)
        np.save(os.path.join(self.dir, f"data_norm_{self.window}_{self.train_test}.npy"), self.data_norm)
        np.save(os.path.join(self.dir, f"mean_{self.window}_{self.train_test}.npy"), self.mean)
        np.save(os.path.join(self.dir, f"std_{self.window}_{self.train_test}.npy"), self.std)

    def _load_ground_truth(self, window, train_test):
        raw_data = torch.from_numpy(np.load(os.path.join(self.dir, f"raw_data_{window}_{train_test}.npy")))
        data_norm = torch.from_numpy(np.load(os.path.join(self.dir, f"data_norm_{window}_{train_test}.npy")))
        mean = torch.from_numpy(np.load(os.path.join(self.dir, f"mean_{window}_{train_test}.npy")))
        std = torch.from_numpy(np.load(os.path.join(self.dir, f"std_{window}_{train_test}.npy")))

        return raw_data, data_norm, mean, std
    
class Stock_Stationary(StationaryDataset):
    def __init__(self,
                 symbol : str = "AAPL, MSFT, NVDA, AMZN, COST", 
                 sdate : str = "20000101", 
                 edate : str = "20231231",
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.5,
                 save_ground_truth : bool =True, 
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/stock_stationary'
        os.makedirs(self.dir, exist_ok=True)

        raw_df = fdr.DataReader(symbol, sdate, edate)
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()
        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_st_norm, self.data, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_st_norm[idx], self.data[idx], self.mean[idx], self.std[idx]

class Stock_NonStationary(NonStationaryDataset):
    def __init__(self,
                 symbol : str = "AAPL, MSFT, NVDA, AMZN, COST", 
                 sdate : str = "20000101", 
                 edate : str = "20231231",
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.5,
                 save_ground_truth : bool =True, 
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/stock_non_stationary'
        os.makedirs(self.dir, exist_ok=True)

        raw_df = fdr.DataReader(symbol, sdate, edate)
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()
        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_norm, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data_norm.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()

    def __len__(self):
        return len(self.data_norm)

    def __getitem__(self, idx):
        return self.data_norm[idx], self.mean[idx], self.std[idx]

class Exchange_Stationary(StationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio: int = 0.7,
                 save_ground_truth : bool =True,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/exchange_stationary'
        os.makedirs(self.dir, exist_ok=True)

        raw_df = pd.read_csv("data/exchange_rate.csv")
        raw_df = raw_df.iloc[:,1:]
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()
        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_st_norm, self.data, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_st_norm[idx], self.data[idx], self.mean[idx], self.std[idx]

class Exchange_NonStationary(NonStationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio: int = 0.7,
                 save_ground_truth : bool =True,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/exchange_non_stationary'
        os.makedirs(self.dir, exist_ok=True)

        raw_df = pd.read_csv("data/exchange_rate.csv")
        raw_df = raw_df.iloc[:,1:]
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()

        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_norm, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data_norm.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()

    def __len__(self):
        return len(self.data_norm)

    def __getitem__(self, idx):
        return self.data_norm[idx], self.mean[idx], self.std[idx]

class ETTh_Stationary(StationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.5,
                 save_ground_truth : bool =True,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/ETTh_stationary'
        os.makedirs(self.dir, exist_ok=True)
        
        raw_df = pd.read_csv("data/ETTh1.csv")
        raw_df = raw_df.drop("date", axis=1)
        raw_df = raw_df[raw_df.columns[::-1]]
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()
        
        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_st_norm, self.data, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_st_norm[idx], self.data[idx], self.mean[idx], self.std[idx]

class ETTh_NonStationary(NonStationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.5,
                 save_ground_truth : bool =True,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/ETTh_non_stationary'
        os.makedirs(self.dir, exist_ok=True)
        
        raw_df = pd.read_csv("data/ETTh1.csv")
        raw_df = raw_df.drop("date", axis=1)
        raw_df = raw_df[raw_df.columns[::-1]]
        raw_data = torch.from_numpy(raw_df.to_numpy()).float()

        self.raw_data = self._train_test_selection(raw_data, train_test)            
        self.data_norm, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
        self.feature_dim = self.data_norm.shape[-1]
        
        if save_ground_truth:
            self._save_ground_truth()

    def __len__(self):
        return len(self.data_norm)

    def __getitem__(self, idx):
        return self.data_norm[idx], self.mean[idx], self.std[idx]

class Energy_Stationary(StationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.7,
                 save_ground_truth : bool =True,
                 download : bool = False,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/energy_stationary'
        os.makedirs(self.dir, exist_ok=True)
        
        if download:
            energy_repo = fetch_ucirepo(id=374)
            raw_df = energy_repo.data.original.drop("date", axis=1)
            raw_df = raw_df.iloc[-10000:,:]
            raw_data = torch.from_numpy(raw_df.to_numpy()).float()
            
            self.raw_data = self._train_test_selection(raw_data, train_test)            
            self.data_st_norm, self.data, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
            
            if save_ground_truth:
                self._save_ground_truth()

        else:
            self.raw_data, self.data_st_norm, self.data, self.mean, self.std = self._load_ground_truth(window, train_test)
            
        self.feature_dim = self.data.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_st_norm[idx], self.data[idx], self.mean[idx], self.std[idx]

class Energy_NonStationary(NonStationaryDataset):
    def __init__(self,
                 window : int = 24,
                 train_test : str ='train',
                 train_ratio : float = 0.7,
                 save_ground_truth : bool =True,
                 download : bool = False,
                 ):
        super().__init__(window, train_test, train_ratio)
        self.dir = './save/energy_non_stationary'
        os.makedirs(self.dir, exist_ok=True)
        
        if download:
            energy_repo = fetch_ucirepo(id=374)
            raw_df = energy_repo.data.original.drop("date", axis=1)
            raw_df = raw_df.iloc[-10000:,:]
            raw_data = torch.from_numpy(raw_df.to_numpy()).float()
            
            self.raw_data = self._train_test_selection(raw_data, train_test)            
            self.data_norm, self.mean, self.std = self._generate_ts_seq_data(self.raw_data, window)
            
            if save_ground_truth:
                self._save_ground_truth()

        else:
            self.raw_data, self.data_norm, self.mean, self.std = self._load_ground_truth(window, train_test)
            
        self.feature_dim = self.data_norm.shape[-1]
    
    def __len__(self):
        return len(self.data_norm)

    def __getitem__(self, idx):
        return self.data_norm[idx], self.mean[idx], self.std[idx]

