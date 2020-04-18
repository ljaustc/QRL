import matplotlib.pyplot as plt
import scipy
import scipy.stats
from models import *


def get_model_color(model_name):
    if 'Base' in model_name:
        color = [0, 0, 0]
    elif 'QSL' in model_name or 'QSPP' in model_name:
        color = [1, 0.5, 0]
    elif 'VPP' not in model_name:
        color = [0.87, 0, 0]
    else:
        color = [0, 0.87, 0.87]
    return color


def plot_start(square=True, figsize=None, ticks_pos=True):
    '''
    unified plot params
    '''
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 6
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 480
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig, ax


def get_group_title(group_name):
    return {
        'con': 'Control Group',
        'smo': 'Smoking Group',
        'stein100': 'Super Group',
    }[group_name]


def get_group_color(group_name):
    return {
        'con': 'C0',
        'smo': 'C1',
        'stein100': 'C2',
    }[group_name]


def get_IC(criterion, finalmle, parnums, data_point_num):
    if criterion == 'AIC':
        IC = 2 * parnums - 2 * finalmle
    elif criterion == 'AICc':
        IC = 2 * parnums - 2 * finalmle + (2 * parnums ** 2 + 2 * parnums) / (data_point_num - parnums - 1)
    elif criterion == 'BIC':
        IC = np.log(data_point_num) * parnums - 2 * finalmle
    return IC


def get_MLE_baseline(ch):
    '''
    compute the MLE for the baseline model
    '''
    choices = np.unique(ch)
    max_pr = np.zeros(ch.shape[0])
    assert len(choices) == 4
    finalmle = np.zeros(ch.shape[0])
    for choice in choices:
        max_pr = np.maximum(max_pr, np.mean(ch == choice, 1))
        finalmle += np.sum(ch == choice, 1) * np.log(np.mean(ch == choice, 1))
    parnums = len(choices) - 1
    return finalmle, parnums, max_pr


def VB_estimation(value):
    '''
    variational Bayesian method
    '''
    def drchrnd(n, a):
        # take a sample from a dirichlet distribution
        p = len(a)
        r = np.zeros([n, p])
        for k in range(p):
            r[:, k] = np.random.gamma(a[k], scale=1, size=n)
        r /= np.sum(r, 1).reshape([-1, 1])
        return r

    [modelnum, subnum] = value.shape
    alpha0 = np.ones(modelnum)
    alpha = alpha0.copy()
    while 1:
        u = np.zeros([modelnum, subnum])
        beta = np.ones(modelnum)
        for n in range(subnum):
            for k in range(modelnum):
                u[k, n] = np.exp(value[k, n] + scipy.special.digamma(alpha[k]) - scipy.special.digamma(np.sum(alpha)))
            u[:, n] /= np.sum(u[:, n])
        for k in range(modelnum):
            beta[k] = np.sum(u[k])
        alpha_new = alpha0 + beta
        if np.mean(abs(alpha_new - alpha)) < 0.0001:
            break
        alpha = alpha_new

    rk = alpha / np.sum(alpha)
    simnum = 5000
    exceed_rand = drchrnd(simnum, alpha)
    exceedPr = np.zeros(modelnum)
    for sim in range(simnum):
        idx = np.argmax(exceed_rand[sim])
        exceedPr[idx] += 1
    exceedPr /= np.sum(exceedPr)
    return alpha, rk, exceedPr


def plot_IC_compare(group_name, criterion, all_model_name):
    '''
    plot AICc/BIC results of all models for one group
    '''
    ch, win, loss, subname = load_subject_data(group_name)
    subnums, data_point_num = ch.shape
    modelnum = len(all_model_name)
    IC = np.zeros([modelnum, subnums])

    for model_idx in range(modelnum):
        model_name = all_model_name[model_idx]
        if model_name == 'Baseline':
            finalmle, parnums, acc = get_MLE_baseline(ch)
        else:
            finalmle, finalpar, _, _, _ = load_fit_data(group_name, model_name)
            parnums = finalpar.shape[1]
        IC[model_idx] = get_IC(criterion, finalmle, parnums, data_point_num)
    [alpha, rk, exceedPr] = VB_estimation(-IC / 2)
    plot_subfigure(IC, all_model_name)
    plt.ylabel(criterion)
    plt.title(get_group_title(group_name))
    plt.xlim(-1, modelnum)
    plt.savefig('figure/' + criterion + '_' + group_name + '.pdf', bbox_inches="tight")
    plt.show()

    plot_subfigure(rk, all_model_name, reverse=True)
    plt.plot([-1, 1 + modelnum], np.ones(2) / modelnum, 'k--')
    plt.ylabel(criterion + '-based Prob.')
    plt.title(get_group_title(group_name))
    plt.yticks([0, 0.4, 0.8])
    plt.ylim([0, 0.8])
    plt.xlim(-1, modelnum)
    plt.savefig('figure/' + criterion + 'Prob_' + group_name + '.pdf', bbox_inches="tight")
    plt.show()


def plot_subfigure(IC, all_model_name, reverse=False, has_text=False):
    model_colors = np.array([get_model_color(model_name) for model_name in all_model_name])
    model_pretty_names = np.array(all_model_name)
    if len(IC.shape) == 2:
        IC_mean = np.mean(IC, 1)
        IC_se = scipy.stats.sem(IC, 1)
    else:
        IC_mean = IC
        IC_se = np.zeros(IC.shape)

    modelnum = len(IC_mean)
    if reverse:
        sort_index = np.argsort(-IC_mean)
    else:
        sort_index = np.argsort(IC_mean)

    plot_start(square=False)
    plt.bar(range(modelnum), IC_mean[sort_index], yerr=IC_se[sort_index], color=model_colors[sort_index])
    if has_text:
        for idx in range(modelnum):
            v = IC_mean[sort_index][idx]
            plt.text(idx + 0.5, v + 0.05, '%.3f' % v, rotation=90, rotation_mode='anchor')
    plt.xticks(np.arange(modelnum), model_pretty_names[sort_index], rotation=90)

if __name__ == '__main__':
    for group_name in ['con', 'smo', 'stein100']:
        for criterion in ['AICc', 'BIC']:
            plot_IC_compare(group_name, criterion, all_paper_model)
