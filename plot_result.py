from json import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

def load_result(filename):
    df = pd.read_csv(filename)
    return df


if __name__=='__main__':

    RES_DIR = 'output'
    filename = f'{RES_DIR}/running_times.csv'

    # Load result
    res_df = load_result(filename)

    # ------------ Plot result ------------

    # Running times
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    X = res_df[res_df['method']=='SpMV']['nb_batch'].array
    plt.plot(X, res_df[res_df['method']=='SpMV']['avg_time_iter'].array, label='SpMV', marker='o')
    plt.plot(X, [res_df[res_df['method']=='skn']['avg_time_iter'].values]*len(X), label='skn', marker='o')
    plt.xlabel('# batches')
    plt.ylabel('Avg time per it (s)')
    plt.legend()
    plt.title('Avg time per it: processing data + PageRank', weight='bold')
    plt.savefig(f'{RES_DIR}/img/running_times')

    # Top10 compared to sknetwork
    scores = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    X = res_df[res_df['method']=='SpMV']['nb_batch'].array
    y_true = set(res_df[res_df['method']=='skn']['top10'].values[0][1:-1].split())
    for idx, row in res_df[res_df['method']=='SpMV'].iterrows():
        vals = row['top10'][1:-1].split()
        top10 = set(vals)
        ratio = len(top10.intersection(y_true)) / len(y_true)
        scores.append(ratio)

    plt.plot(X, scores, label='HIT ratio', marker='o')
    plt.xlabel('# batches')
    plt.ylabel('HIT ratio')
    plt.legend()
    plt.title('HIT@10 ratio compared to scikit-network result', weight='bold')
    plt.savefig(f'{RES_DIR}/img/HIT_ratio.png')
