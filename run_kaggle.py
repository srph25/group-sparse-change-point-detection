import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import gc
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed
import weighted_gflasso



powers_lmbd = list(range(-10, 14, 1))
powers_thres = list(range(-10, 14, 1))


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18


def plot_series(X, M, Y, lmbd, p, mode, img_path):
    fig, axs = plt.subplots(X.shape[1], sharex=True)
    title = 'Kaggle SFDDD Data Set Example\n'
    if mode in ['weighted', 'unweighted']:
        if mode == 'weighted':
            title += 'Weighted '
        if p == 2:
            title += 'Group '
        title += 'Fused LASSO, ' + r'$\lambda={{{}}}$'.format(lmbd)
        ylabel = 'Joint Coordinates [Pixel Index]'
    elif mode == 'weights':
        title += 'Input Weights'
        ylabel = 'Joint Weights [Dimensionless]'
    fig.suptitle(title, y=1.0005)
    for i in range(X.shape[1]):
        if i == 0:
            if mode in ['weighted', 'unweighted']:
                axs[i].plot(X[:, i], 'b-', label="Input")
                axs[i].plot(Y[:, i], 'r-', label="Output")
            else:
                axs[i].plot(M[:, i], '-', color='black', label="Input Weights")
                axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        else:
            if mode in ['weighted', 'unweighted']:
                axs[i].plot(X[:, i], 'b-', label='_nolegend_')
                axs[i].plot(Y[:, i], 'r-', label='_nolegend_')
            else:
                axs[i].plot(M[:, i], '-', color='black', label='_nolegend_')
                axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))            
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [Video Frame Index]")
    gca = plt.gca()
    gca.xaxis.set_label_coords(0.5, -0.085)
    plt.ylabel(ylabel)
    if mode in ['weighted', 'unweighted']:
        fig.legend(loc=(0.67, 0.73))
    else:
        fig.legend(loc=(0.57, 0.8025))
    #plt.show()
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
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
        title = 'Kaggle SFDDD Data Set Example\nOutput Change-point Scores'
        fig.suptitle(title, y=1.0005)
        ylabel = 'Output Euclidean Norm [Pixel Index]'
        ax.plot(np.arange(len(score)), score, 'r-', label="Norm of first derivative")
        ax.plot(np.arange(len(score)), thres * np.ones_like(score), 'r--', label="Threshold")
        for idx, t in enumerate(np.where(true == 1)[0]):
            if idx == 0:
                ax.plot(t, 0, 'g.', markersize=14, label="Ground truth change-point")
            else:
                ax.plot(t, 0, 'g.', markersize=14, label='_nolegend_')
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time [Video Frame Index]")
        gca = plt.gca()
        gca.xaxis.set_label_coords(0.5, -0.085)
        plt.ylabel(ylabel)
        ax.legend(loc=(0.3, 0.72))
        #plt.show()

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        fig.savefig(img_path, dpi=600)
        #fig.clear()
        plt.close(fig)
        del fig, ax
        gc.collect()


if __name__ == "__main__":
    cwd = os.getcwd()
    arm_dir = os.path.join(cwd, 'kaggle/pose_features')
    label_dir = os.path.join(cwd, 'kaggle/labels')
    results_dir = os.path.join(cwd, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_dir = os.path.join(cwd, 'results/kaggle')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    roc_auc = {}
    f1 = {}
    for file_name in np.sort(os.listdir(label_dir)):
        base_file_name = file_name[:18]
        label_path = os.path.join(label_dir, file_name)
        arm_path = os.path.join(arm_dir, base_file_name + '_arm.npy.npz')
        if os.path.isfile(label_path) is True:
            print(file_name[:18])
            labels = np.load(label_path, allow_pickle=True, encoding = 'latin1')
            true = np.diff(labels)
            arm_data = np.load(arm_path, allow_pickle=True, encoding = 'latin1')
            X = np.reshape(arm_data['poses'], (arm_data['poses'].shape[0], -1))
            M = np.reshape(arm_data['scores'], (arm_data['scores'].shape[0], -1))
            assert(labels.shape[0] == X.shape[0] and labels.shape[0] == M.shape[0])
            print(labels.shape, X.shape, M.shape)

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

