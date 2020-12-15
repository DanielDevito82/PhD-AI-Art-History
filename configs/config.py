# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path": "oxford_iiit_pet:3.*.*",
        "path_labels_csv": 'C:\\Users\\mestrovicd\\Desktop\\Wissen\\Deep_Learning\\TF_2_Notebooks_and_Data\\Project\\Images\\features.csv',
        "csv_sep": ";",
        "image_size": 256,
        "folder": 'C:\\Users\\mestrovicd\\Desktop\\Wissen\\Deep_Learning\\TF_2_Notebooks_and_Data\\Project\\Images\\_trimed\\_1'
    },
    "train": {
        "batch_size": 32,
        "buffer_size": 1000, # brauch ist das?
        "epoches": 100,
        "val_subsplits": 5, # brauch ist das?
        "optimizer": {
            "type": "adam"
        },
        "EarlyStopping": {
            "monitor": "val_loss",
            "mode": "auto",
            "patience": 3
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": (256, 256, 1),
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 5
    }
}
