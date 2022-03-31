import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import time
from scipy import sparse
from line_profiler import LineProfiler

from sknetwork.ranking import PageRank
import matplotlib.pyplot as plt

def count_neighbors(edge_file):
    outdeg_orig = {}
    outdeg = {}
    reindex = {}
    idx = 0

    if os.path.exists(f'data/neighbs_swh_{edge_file}.csv') and os.path.exists(f'data/reindex_{edge_file}.csv'):
        with open(f'data/reindex_{edge_file}.csv') as f:
            for line in tqdm(f):
                vals = line.strip('\n').split(',')
                reindex[vals[0]] = int(vals[1])
        with open(f'data/neighbs_swh_{edge_file}.csv') as f:
            for line in tqdm(f):
                vals = line.strip('\n').split(',')
                outdeg[int(vals[0])] = int(vals[1])
    else:
        with open(f'data/{edge_file}.txt') as f:
            with open(f'data/{edge_file}.csv', 'w') as o:
                for i, line in enumerate(tqdm(f)):
                    if i != 0:
                        vals = line.strip('\n').split('\t')
                        src, dst = vals[0], vals[1]

                        # reindex nodes
                        if src not in reindex:
                            reindex[src] = idx
                            idx += 1
                        if dst not in reindex:
                            reindex[dst] = idx
                            idx += 1

                        # count outdegrees
                        #outdeg_orig[src] = outdeg_orig.get(src, 0) + 1
                        #outdeg_orig[dst] = outdeg_orig.get(dst, 0) + 1
                        outdeg[reindex[src]] = outdeg.get(reindex[src], 0) + 1
                        outdeg[reindex[dst]] = outdeg.get(reindex[dst], 0) + 1   

                        o.write(f'{reindex[src]},{reindex[dst]}\n')  
        
        with open(f'data/neighbs_swh_{edge_file}.csv', 'w') as f:
            #f.writelines(f'{k},{v}\n' for k, v in outdeg_orig.items())
            f.writelines(f'{k},{v}\n' for k, v in outdeg.items())
        with open(f'data/reindex_{edge_file}.csv', 'w') as f:
            f.writelines(f'{k},{v}\n' for k, v in reindex.items())

    #raise Exception('end')
    return outdeg, reindex


def main():
    TOY = False
    data_dir = 'data'
    if TOY:
        filename = 'soc-LiveJournal1_toy' #'soc-LiveJournal1_toy'
    else:
        filename = 'soc-LiveJournal1' #'soc-LiveJournal1_toy'
    ext = 'txt'   

    # Preprocess data (if needed)
    """print('---------------------')
    print('Preprocessing')
    print('---------------------')

    print('Load edges...')
    #_ = preprocess(edges)    
    #edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], skiprows=1, delimiter=',')
    edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter='\t')
    #edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter=',')
    print('Count outdegrees...')
    neighbs_count, reindex = count_neighbors(f'{filename}')
    print(f'Number neighbors count begin: {len(neighbs_count)}')
    print(f'Number reindex count begin: {len(reindex)}')

    print('---------------------')
    print('PageRank')
    print('---------------------')

    m = edges.shape[0]
    n = len(neighbs_count)
    batch_size = int(1 * m)
    labels = {} # dictionary swhid: idx

    scores_batches = []
    nb_iterations = 5

    init_scores = []
    print('Load edges reindexed...')
    edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], delimiter=',')
        
    with open(f'output/top10_seq_{filename}_{batch_size}.txt', 'w') as output:
        
        for it in range(nb_iterations):
            batches = (edges[pos:pos+batch_size] for pos in range(0, len(edges), batch_size))

            start_it = time.time()
            if len(init_scores) > 0:
                scores = init_scores.copy()
            else:
                scores = np.ones(n) / n

            for idx, batch in enumerate(batches):
                min_pr = (1 - 0.85) / n
                scores_ = np.zeros(n)
                for src, dst in tqdm(zip(batch['src'], batch['dst'])):
                    #src = reindex.get(src)
                    #dst = reindex.get(dst)
                    scores_[dst] += 0.85 * scores[src] / neighbs_count.get(src)
                    scores_[src] += 0.85 * scores[dst] / neighbs_count.get(dst)

                #print('update: ', scores_ + min_pr)
                scores = scores_ + scores + min_pr
                #print('scores at end of iteration : ', scores)
                scores = scores / scores.sum()
                
                init_scores = scores.copy()
            print(f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.')
            output.write((f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.\n'))

        # Display result
        print(f'Top 10 project: {np.argsort(-scores)[:10]}')
        top10_idx = np.argsort(-scores)[:10]
        keys = np.array(list(reindex.keys()))
        values = np.array(list(reindex.values()))
        
        # Save results
        output.write(f'Top 10 project: {np.argsort(-scores)[:10]}\n')
        for i in top10_idx:
            print(keys[np.where(values==i)[0]][0], scores[i])
            output.write(f'{keys[np.where(values==i)[0]][0]} {scores[i]}\n')"""



    print('---------------------')
    print('Preprocessing vect prod')
    print('---------------------')

    columns = ['method', 'nb_batch', 'avg_time_iter', 'top10']
    res_df = pd.DataFrame(columns=columns)

    print('Load edges...')
    #_ = preprocess(edges)    
    #edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], skiprows=1, delimiter=',')
    if 'toy' in filename:
        edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter=',')
    else:
        edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter='\t')
    print('Count outdegrees...')
    neighbs_count, reindex = count_neighbors(f'{filename}')
    print(f'Number neighbors count begin: {len(neighbs_count)}')
    print(f'Number reindex count begin: {len(reindex)}')
    
    m = edges.shape[0]
    n = len(neighbs_count)
    if 'toy' in filename:
        batch_size = int(1 * m)
    else:
        batch_size = int(1 * m)
    labels = {} # dictionary swhid: idx

    scores_batches = []
    nb_iterations = 5

    init_scores = np.zeros(n)
    UNDIRECTED = True

    # Out degree matrix
    out_degrees = np.array(list(dict(sorted(neighbs_count.items(), key=lambda x: x[0], reverse=False)).values()))
    diag: sparse.coo_matrix = sparse.diags(out_degrees, format='coo')
    diag.data = 1 / diag.data
    print(f'max out degree: {np.max(out_degrees)}') 
    print('corresponding index: ', np.where(out_degrees==np.max(out_degrees)))
    print('corresponding original nodes: ', np.array(list(reindex.keys()))[np.where(out_degrees==np.max(out_degrees))][0])

    print('---------------------')
    print('PageRank vect prod')
    print('---------------------')

    print('Load edges reindexed...')
    edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], delimiter=',')

    batch_sizes = [int(1*m), int(0.5*m), int(0.1*m), int(0.05), int(0.01*m), int(0.005*m)]
    times_vect_prod = []

    for batch_size in batch_sizes:

        times_vect_prod_iter = []
        with open(f'output/top10_SpMv{filename}_{batch_size}.txt', 'w') as output:

            for it in range(nb_iterations):
                batches = (edges[pos:pos+batch_size] for pos in range(0, len(edges), batch_size))
                
                start_it = time.time()
                
                if np.sum(init_scores) != 0:
                    # sequential scenario
                    scores = init_scores.copy()
                    # parallel scenario
                    #scores = init_scores.sum(axis=0)
                    #init_scores = scores.copy()
                else:
                    scores = np.ones(n) / n
                

                for idx, batch in tqdm(enumerate(batches)):
                    #print('-------------- \nBatch ', idx)

                    # Adjacency matrix
                    #max_idx = max(np.max(batch['src'].array), np.max(batch['dst'].array)) + 1
                    if UNDIRECTED:
                        rows = np.hstack([batch['src'].array, batch['dst'].array])
                        cols = np.hstack([batch['dst'].array, batch['src'].array])
                        data = np.ones(len(rows))
                    else:
                        rows = batch['src'].array
                        cols = batch['dst'].array
                        data = np.ones(len(batch['src']))
                    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
                    
                    # Matrices
                    #W = (0.85 * diag.tocsr()[:max_idx, :max_idx].dot(adjacency)).T.tocoo()
                    W = (0.85 * diag.dot(adjacency)).T.tocoo()
                    v0 = (np.ones(adjacency.shape[1]) - 0.85) / adjacency.shape[1]

                    # SpMV multiplication
                    scores_ = W.dot(scores) + v0 #+ np.ones(len(scores))/len(scores)
                    scores_ /= scores_.sum()
                    #init_scores = np.vstack([init_scores, scores_]) # mimic the case of a parallel scenario
                    init_scores += scores + scores_ # mimic the case of a sequential scenario

                    #print('new init scores: ', init_scores)
                    #print('scores_: ', scores_)
                print (f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.')
                output.write((f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.\n'))
                times_vect_prod_iter.append(time.time() - start_it)

            #scores = init_scores.sum(axis=0) / np.sum(init_scores.sum(axis=0)) # parallel scenario
            scores = init_scores / np.sum(init_scores) # sequential scenario

            # Display result
            print(f'Top 10 project: {np.argsort(-scores)[:10]}')
            output.write(f'Top 10 project: {np.argsort(-scores)[:10]}\n')
            top10_idx = np.argsort(-scores)[:10]
            keys = np.array(list(reindex.keys()))
            values = np.array(list(reindex.values()))
            for i in top10_idx:
                print(keys[np.where(values==i)[0]][0], scores[i])
                output.write(f'{keys[np.where(values==i)[0]][0]} {scores[i]}\n')

            # Save variables in DataFrame
            times_vect_prod = np.mean(times_vect_prod_iter)
            tmp_df = pd.DataFrame([['SpMV', m/batch_size, times_vect_prod, np.argsort(-scores)[:10]]], columns=columns)
            res_df = res_df.append(tmp_df, ignore_index=True)
            print(res_df.head())

    print('---------------------')
    print('Scikit-network')
    print('---------------------')

    print('Load edges...')
    #_ = preprocess(edges)    
    #edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], skiprows=1, delimiter=',')
    if 'toy' in filename:
        edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter=',')
    else:
        edges = pd.read_csv(f'{data_dir}/{filename}.txt', names=['src', 'dst'], skiprows=1, delimiter='\t')
    print('Count outdegrees...')
    neighbs_count, reindex = count_neighbors(f'{filename}')
    print(f'Number neighbors count begin: {len(neighbs_count)}')
    print(f'Number reindex count begin: {len(reindex)}')
    
    m = edges.shape[0]
    n = len(neighbs_count)
    if 'toy' in filename:
        batch_size = int(1 * m)
    else:
        batch_size = int(1 * m)
    labels = {} # dictionary swhid: idx

    scores_batches = []
    nb_iterations = 1

    init_scores = np.zeros(n)
    UNDIRECTED = True

    # Out degree matrix
    out_degrees = np.array(list(dict(sorted(neighbs_count.items(), key=lambda x: x[0], reverse=False)).values()))
    diag: sparse.coo_matrix = sparse.diags(out_degrees, format='coo')
    diag.data = 1 / diag.data
    print(f'max out degree: {np.max(out_degrees)}') 
    print('corresponding index: ', np.where(out_degrees==np.max(out_degrees)))
    print('corresponding original nodes: ', np.array(list(reindex.keys()))[np.where(out_degrees==np.max(out_degrees))][0])

    print('---------------------')
    print('PageRank scikit-network')
    print('---------------------')

    print('Load edges reindexed...')
    edges = pd.read_csv(f'{data_dir}/{filename}.csv', names=['src', 'dst'], delimiter=',')

    with open(f'output/top10_skn_{filename}_{batch_size}.txt', 'w') as output:

        for it in range(nb_iterations):
            batches = (edges[pos:pos+batch_size] for pos in range(0, len(edges), batch_size))
            start_it = time.time()

            for idx, batch in tqdm(enumerate(batches)):
                if UNDIRECTED:
                    rows = np.hstack([batch['src'].array, batch['dst'].array])
                    cols = np.hstack([batch['dst'].array, batch['src'].array])
                    data = np.ones(len(rows))
                else:
                    rows = batch['src'].array
                    cols = batch['dst'].array
                    data = np.ones(len(batch['src']))

                adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
                print('\nCOO to CSR format...')
                adjacency_csr = adjacency.tocsr()
                pagerank = PageRank()
                print('Run PageRank...')
                scores = pagerank.fit_transform(adjacency_csr)

        print (f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.')
        output.write((f'Completed {it+1:>3}/{nb_iterations} iterations. {time.time() - start_it:.5f} seconds elapsed.\n'))

        # Display result
        print(f'Top 10 project: {np.argsort(-scores)[:10]}')
        output.write(f'Top 10 project: {np.argsort(-scores)[:10]}\n')
        top10_idx = np.argsort(-scores)[:10]
        keys = np.array(list(reindex.keys()))
        values = np.array(list(reindex.values()))
        for i in top10_idx:
            print(keys[np.where(values==i)[0]][0], scores[i])
            output.write(f'{keys[np.where(values==i)[0]][0]} {scores[i]}\n')

        # Save variables to DataFrame
        times_skn = (time.time() - start_it) / 10
        tmp_df = pd.DataFrame([['skn', 1, times_skn, np.argsort(-scores)[:10]]], columns=columns)
        res_df = res_df.append(tmp_df, ignore_index=True)

    # saving the dataframe
    res_df.to_csv('output/running_times.csv')

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.plot(res_df[res_df['method']=='SpMV']['nb_batch'].array, res_df[res_df['method']=='SpMV']['avg_time_iter'].array, label='SpMV', marker='o')
    plt.plot(res_df[res_df['method']=='SpMV']['nb_batch'].array, [res_df[res_df['method']=='skn']['avg_time_iter'].values]*len(batch_sizes), label='skn', marker='o')
    plt.xlabel('# batches')
    plt.ylabel('Avg time per it (s)')
    plt.legend()
    plt.title('Avg time per it: processing data + PageRank', weight='bold')
    plt.savefig('output/img/running_times')
    #plt.show()


# Line profiler
lprofiler = LineProfiler()
lp_wrapper = lprofiler(main)
lp_wrapper()
#lprofiler.print_stats()
