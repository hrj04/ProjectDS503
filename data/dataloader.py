import torch
from utils.utils import instantiate_from_config

def dataloader_info(config, train=True):
    batch_size = config['dataloader']['batch_size']
    if train :
        dataset = instantiate_from_config(config['dataloader']['train_dataset'])
        jud = config['dataloader']['train_dataset']['shuffle']

    else : 
        dataset = instantiate_from_config(config['dataloader']['test_dataset'])
        jud = config['dataloader']['test_dataset']['shuffle']

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)
    dl_info = {'dataloader': dataloader,
               'dataset': dataset}

    return dl_info
