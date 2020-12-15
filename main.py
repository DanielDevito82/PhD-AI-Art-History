# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.MyModel import MyModel
from dataloader import dataloader
import pandas as pd


def run():
    """Builds model, loads data, trains and evaluates"""
    #df = dataloader.DataLoader().load_data(CFG)
    #print(df[2])
    model = MyModel(CFG)
    model.load_data()
    model.build()
    metrics = pd.DataFrame(model.train())
    print(metrics)
    # model.evaluate()


if __name__ == '__main__':
    run()
