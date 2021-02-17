import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg') 
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed
from scipy.stats import truncnorm
import weighted_gflasso


np.random.seed(1234)
m = 50 # series
n = 500 # time steps
d = 3 # features
p = 4 # steps
powers_lmbd = list(range(-10, 14, 1))
powers_thres = list(range(-10, 14, 1))


def plot_series(X, M, Y, lmbd, p, mode, img_path):
    fig, axs = plt.subplots(X.shape[1], sharex=True)
    if mode in ['weighted', 'unweighted']:
        title = ''
        if mode == 'weighted':
            title += 'Weighted '
        if p == 2:
            title += 'Group '
        title += 'Fused LASSO Denoising\n' + r'$\lambda={{{}}}$'.format(lmbd)
        fig.suptitle(title)
        ylabel = 'Features [Dimensionless]'
    elif mode == 'weights':
        fig.suptitle('Weights')
        ylabel = 'Weights [Dimensionless]'
    for i in range(X.shape[1]):
        if mode in ['weighted', 'unweighted']:
            axs[i].plot(X[:, i], 'b-', label="Input")
            axs[i].plot(Y[:, i], 'r-', label="Output")
        else:
            axs[i].plot(M[:, i], 'g-', label="Input Weights")
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [Index]")
    plt.ylabel(ylabel)
    axs[0].legend(loc="upper right")
    #plt.show()
    fig.savefig(img_path, dpi=600)
    #fig.clear()
    plt.close(fig)
    del fig, axs
    gc.collect()


def plot_labels(score, pred, true, thres, lmbd, p, mode, img_path):
    if mode == 'weights':
        pass
    elif mode in ['weighted', 'unweighted']:
        fig, ax = plt.subplots(1, sharex=True)
        title = 'Output Gross-but-'
        if p == 2:
            title += 'Group-'
        title += 'Sparse Norm Values\n of '
        if mode == 'weighted':
            title += 'Weighted '
        if p == 2:
            title += 'Group '
        title += 'Fused LASSO Denoising, ' + r'$\lambda={{{}}}$'.format(lmbd)
        fig.suptitle(title)
        ylabel = 'Output Euclidean Norm [Dimensionless]'
        label = "Norm of gross-but-"
        if p == 2:
            label += "group-"
        label += "sparse first derivative"
        ax.plot(np.arange(len(score)), score, 'b-', label=label)
        ax.plot(np.arange(len(score)), thres * np.ones_like(score), 'r--', label="Threshold for predicted change-point label")
        for idx, t in enumerate(np.where(true == 1)[0]):
            if idx == 0:
                ax.plot(t, 0, 'g.', markersize=14, label="Ground truth logged time of change-point")
            else:
                ax.plot(t, 0, 'g.', markersize=14, label='_nolegend_')
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [Index]")
        plt.ylabel(ylabel)
        ax.legend(loc="upper right")
        #plt.show()
        fig.savefig(img_path, dpi=600)
        #fig.clear()
        plt.close(fig)
        del fig, ax
        gc.collect()


if __name__ == "__main__":
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_dir = os.path.join(cwd, 'results/toy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    roc_auc = {}
    f1 = {}
    for q in range(m):
        print(q)
        base_file_name = ('series_' + str(q))

        x = np.arange(n)
        idx = np.random.choice(n - 1, p - 1)
        diff = np.zeros((n - 1, d))
        true = np.zeros((n - 1,))
        diff[idx, :] = np.random.randn(p - 1, d)
        true[idx] = 1
        init = np.random.randn(d)
        X = np.concatenate([init[None, :], init[None, :] + np.cumsum(diff, axis=0)], axis=0)
        std = truncnorm.rvs(0, 3, size=(n, d))
        noise = np.random.randn(*X.shape)
        X += (std * noise)
        M = (1 / std)
        assert(X.shape[0] == (true.shape[0] + 1) and X.shape == M.shape)
        print(true.shape, X.shape, M.shape)

        def run(lmbd):
            roc_auc_lmbd = {}
            f1_lmbd = {}
            for weighted in [True, False]:
                for p in [1, 2]:
                    full_file_name = base_file_name + '_res_' + str(lmbd) + '_' + str(weighted) + '_' + str(p) 
                    save_path = os.path.join(save_dir, full_file_name + '.npy')
                    score_path = os.path.join(save_dir, full_file_name + '_score.npy')
                    if not os.path.exists(save_path): # continue from last checkpoint
                        Y = weighted_gflasso.run(X=X, lmbd=lmbd, M=M if weighted is True else None, norm_type=p)
                        np.save(save_path, Y)
                    else:
                        Y = np.load(save_path)
                    if not os.path.exists(score_path):
                        score = np.linalg.norm(np.diff(Y, axis=0), ord=p, axis=1)
                        np.save(score_path, score)
                    else:                        
                        score = np.load(score_path)
                    roc_auc_lmbd[(lmbd, weighted, p)] = roc_auc_score(true, score)
                    
                    img_path = os.path.join(save_dir, full_file_name  + '_series.pdf')
                    if not os.path.exists(img_path): # continue from last checkpoint
                        if weighted is True:
                            plot_series(X, M, Y, lmbd, p=p, mode='weighted', img_path=img_path)
                            if p == 1:
                                img_path2 = os.path.join(save_dir, base_file_name + '_res_' + str(lmbd) + '_weights.pdf')
                                if not os.path.exists(img_path2): # continue from last checkpoint
                                    plot_series(X, M, Y, lmbd, p=p, mode='weights', img_path=img_path2)
                        else:
                            plot_series(X, M, Y, lmbd, p=p, mode='unweighted', img_path=img_path)

                    for power_thres in powers_thres:
                        thres = 2**power_thres
                        pred_path = os.path.join(save_dir, full_file_name + '_' + str(thres) + '_pred.npy')
                        pred = (score >= thres)
                        np.save(pred_path, pred)
                        f1_lmbd[(lmbd, weighted, p, thres)] = f1_score(true, pred)
                        
                        img_path3 = os.path.join(save_dir, full_file_name  + '_' + str(thres) + '_labels.pdf')
                        if not os.path.exists(img_path3): # continue from last checkpoint
                            if weighted is True:
                                plot_labels(score, pred, true, thres, lmbd, p=p, mode='weighted', img_path=img_path3)
                            else:
                                plot_labels(score, pred, true, thres, lmbd, p=p, mode='unweighted', img_path=img_path3)
            return roc_auc_lmbd, f1_lmbd

        results = Parallel(n_jobs=14)(delayed(run)(lmbd=2**power_lmbd) for power_lmbd in powers_lmbd)
        for (power_lmbd, (roc_auc_lmbd, f1_lmbd)) in zip(powers_lmbd, results):
            lmbd = 2**power_lmbd
            for weighted in [True, False]:
                for p in [1, 2]:
                    print('    lambda=', lmbd, ', weighted=', weighted, ', p=', p)
                    print('        ROC AUC=', roc_auc_lmbd[(lmbd, weighted, p)])
                    for power_thres in powers_thres:
                        thres = 2**power_thres
                        print('        thres=', thres, ', F1=', f1_lmbd[(lmbd, weighted, p, thres)])
            for (k, v) in roc_auc_lmbd.items():
                if k not in list(roc_auc.keys()):
                    roc_auc[k] = []
                roc_auc[k].append(v)
            for (k, v) in f1_lmbd.items():
                if k not in list(f1.keys()):
                    f1[k] = []
                f1[k].append(v)



    for weighted in [True, False]:
        for p in [1, 2]:
            print(weighted, p)

            k_max_roc_auc = None
            v_max_roc_auc = None
            for (k, v) in roc_auc.items():
                if k[1] == weighted and k[2] == p:
                    if k_max_roc_auc is None and v_max_roc_auc is None:
                        k_max_roc_auc = k
                        v_max_roc_auc = np.mean(v)
                    else:
                        if np.mean(v) > v_max_roc_auc:
                            k_max_roc_auc = k
                            v_max_roc_auc = np.mean(v)
            print('    Best mean ROC AUC score:')
            print('        lambda=', k_max_roc_auc[0], ' : ', v_max_roc_auc, '+-', np.std(roc_auc[k_max_roc_auc]),
                           '(F1 score: ', np.max([np.mean(f1[k_max_roc_auc + (2**power_thres,)]) for power_thres in powers_thres]), 
                           '+-', np.std(f1[k_max_roc_auc + (2**powers_thres[np.argmax([np.mean(f1[k_max_roc_auc + (2**power_thres,)]) for power_thres in powers_thres])],)]), ')')

            k_max_f1 = None
            v_max_f1 = None
            for (k, v) in f1.items():
                if k[1] == weighted and k[2] == p:
                    if k_max_f1 is None and v_max_f1 is None:
                        k_max_f1 = k
                        v_max_f1 = np.mean(v)
                    else:
                        if np.mean(v) > v_max_f1:
                            k_max_f1 = k
                            v_max_f1 = np.mean(v)
            print('    Best mean F1 score:')
            print('        lambda=', k_max_f1[0], ', thres=', k_max_f1[3], ' : ', v_max_f1, '+-', np.std(f1[k_max_f1]),
                           '(ROC AUC score: ', np.mean(roc_auc[k_max_f1[:3]]), '+-', np.std(roc_auc[k_max_f1[:3]]), ')')

