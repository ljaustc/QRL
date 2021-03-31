import numpy as np
from scipy.io import loadmat

all_paper_model = ['QSL', 'QSPP', 'PVLDecayTIC', 'PVLDeltaTIC', 'VPPDecayTIC', 'VPPDeltaTIC', 'PVLDecayTDC',
                   'PVLDeltaTDC', 'EVLDecayTIC', 'EVLDeltaTIC', 'EVLDecayTDC', 'EVLDeltaTDC', 'VPPDecayTDC',
                   'VPPDeltaTDC', 'Baseline']
all_group_name = ['con', 'smo', 'stein100']


def expect_utility(win, loss, W):
    return (1 - W) * win - W * np.abs(loss)


def prospect_utility(payoff, alpha, winlambda, losslambda):
    if payoff >= 0:
        return winlambda * np.abs(payoff) ** alpha
    else:
        return -losslambda * np.abs(payoff) ** alpha


def TIC(V, cons):
    return V * (3 ** cons - 1)


def TDC(V, cons, trial):
    return V * (trial / 10) ** cons


def cal_epi(payoff, eP, eN):
    if payoff >= 0:
        return eP
    else:
        return eN


def convert_Pr(theta):
    theta = theta - (np.max(theta) - 20)
    theta = np.exp(theta)
    Pr = theta / np.sum(theta)
    return Pr


def Decay(V, utility, choice, A):
    V = V.copy()
    V = V * A
    V[choice] = V[choice] + utility
    return V


def Delta(V, utility, choice, A):
    V = V.copy()
    PE = utility - V[choice]
    V[choice] = V[choice] + A * (utility - V[choice])
    return V, PE


def amp2Pr(amp):
    '''
    compute probability distribution from amplitudes
    '''
    if len(amp.shape) == 1:
        amp = amp.reshape([len(amp), 1])
    assert amp.shape[1] == 1
    Pr = np.real(amp * amp.conj())
    return Pr


def Grover(phi1, phi2, amp, choice):
    if len(amp.shape) == 1:
        amp = amp.reshape([len(amp), 1])
    assert amp.shape[1] == 1
    N = amp.shape[0]
    Ak = np.zeros([N, N]).astype(np.complex128)
    Ak[choice, choice] = 1 - np.exp(1j * phi1)
    Q1 = np.eye(N) - Ak
    Q2 = (1 - np.exp(1j * phi2)) * np.dot(amp, amp.conj().T) - np.eye(N)
    U_Grover = np.dot(Q2, Q1)
    amp = np.dot(U_Grover, amp)
    amp = amp / np.linalg.norm(amp)
    return amp, U_Grover


def get_Q_Pr_stable(thetaindex, bias1, bias2, amp, utility, choice):
    theta = thetaindex * np.pi
    phi1 = np.pi * (np.cos(theta) * utility + bias1)
    phi2 = np.pi * (np.sin(theta) * utility + bias2)
    amp, _ = Grover(phi1, phi2, amp, choice)
    return amp, amp2Pr(amp), phi1, phi2


def Pr2Entropy(Pr):
    Pr = Pr.copy()
    Pr[Pr < 10 ** -8] = 10 ** -9
    logPr = np.log(Pr)
    Pr[Pr < 10 ** -8] = 0  # keep the results same as matlab
    ent = -np.sum(Pr * logPr)

    return ent


class WorkMemory(object):
    def __init__(self):
        self.ev = np.zeros([4, 1])
        self.P = np.zeros([4, 1])
        self.Pr = np.ones([4, 1]) / 4
        self.amp = np.ones([4, 1]) / 2 * (1 + 0j)
        self.ent = Pr2Entropy(self.Pr)
        self.trial = 0
        self.rho = np.dot(self.amp, self.amp.conj().T)


class Parameters(object):
    def __init__(self):
        pass


class Model(object):

    def __init__(self, X, init_state=None):
        self.wm = WorkMemory()
        self.par = Parameters()
        self.payscale = 100
        X = np.insert(X, 0, 0)
        self.init_par(X)
        if init_state is not None:
            self.set_state(init_state)

    def init_par(self, X):
        pass

    def set_state(self, init_state):
        pass

    def __update__(self, choice, win, loss, trial):
        '''
        this func is called in each trial
        defined in each individual model
        '''
        pass

    def update(self, choice, win, loss, trial):
        # at first(0 trial), no input of choice, return Prob & Ent for t=1 trial
        # input of choice at t trial, return utility & MLE at t trial, and Prob & Ent for t+1 trial
        # choice-=1
        assert choice >= 0 and choice < 4
        self.wm.win = win / self.payscale
        self.wm.loss = np.abs(loss) / self.payscale
        self.wm.payoff = (win - np.abs(loss)) / self.payscale
        self.wm.MLE1trial = np.log(self.wm.Pr[choice, 0] + 1e-20)
        self.wm.rdelta = np.zeros([4, 1])
        self.wm.rdelta[choice] = 1
        self.__update__(choice, win, loss, trial)
        self.wm.trial = trial
        self.wm.ent = Pr2Entropy(self.wm.Pr)


class PVL_init(Model):
    def init_par(self, X):
        self.par_num = 4
        assert len(X) == self.par_num + 1
        self.par.A = X[1]
        self.par.alpha = X[2]
        self.par.cons = X[3]
        self.par.losslambda = X[4]


class EVL_init(Model):
    def init_par(self, X):
        self.par_num = 3
        assert len(X) == self.par_num + 1
        self.par.A = X[1]
        self.par.W = X[2]
        self.par.cons = X[3]


class VPP_init(Model):
    def init_par(self, X):
        self.par_num = 8
        assert len(X) == self.par_num + 1
        self.par.A = X[1]
        self.par.alpha = X[2]
        self.par.cons = X[3]
        self.par.losslambda = X[4]
        self.par.k = X[5]
        self.par.eP = X[6]
        self.par.eN = X[7]
        self.par.w = X[8]


class QSL(Model):
    def init_par(self, X):
        self.par_num = 6
        assert len(X) == self.par_num + 1
        self.par.thetaindex = X[1]
        self.par.alpha = X[2]
        self.par.winlambda = X[3]
        self.par.losslambda = X[4]
        self.par.bias1 = X[5]
        self.par.bias2 = X[6]

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, self.par.winlambda, self.par.losslambda)
        pre_amp = self.wm.amp
        [self.wm.amp, self.wm.Pr, self.wm.phi1, self.wm.phi2] = get_Q_Pr_stable(self.par.thetaindex, self.par.bias1,
                                                                                self.par.bias2, self.wm.amp,
                                                                                self.wm.utility, choice)
        amp, self.wm.U_Grover = Grover(self.wm.phi1, self.wm.phi2, pre_amp, choice)


class QSPP(Model):
    def init_par(self, X):
        self.par_num = 10
        assert len(X) == self.par_num + 1
        self.par.thetaindex = X[1]
        self.par.alpha = X[2]
        self.par.winlambda = X[3]
        self.par.losslambda = X[4]
        self.par.bias1 = X[5]
        self.par.bias2 = X[6]
        self.par.k = X[7]
        if X[8] >= 0:
            self.par.eP = 3 ** X[8] - 1
        else:
            self.par.eP = -3 ** -X[8] + 1
        if X[9] >= 0:
            self.par.eN = 3 ** X[9] - 1
        else:
            self.par.eN = -3 ** -X[9] + 1

        self.par.w = X[10]

    def __update__(self, choice, win, loss, trial):

        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, self.par.winlambda, self.par.losslambda)
        self.wm.epi = cal_epi(self.wm.payoff, self.par.eP, self.par.eN)
        [self.wm.amp, self.wm.PQ, self.wm.phi1, self.wm.phi2] = get_Q_Pr_stable(self.par.thetaindex, self.par.bias1,
                                                                                self.par.bias2, self.wm.amp,
                                                                                self.wm.utility, choice)
        self.wm.P = Decay(self.wm.P, self.wm.epi, choice, self.par.k)
        self.wm.PA = convert_Pr(self.wm.P)
        self.wm.Pr = self.par.w * self.wm.PQ + (1 - self.par.w) * self.wm.PA


class PVLDecayTIC(PVL_init, Model):

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TIC(self.wm.ev, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class PVLDeltaTIC(PVL_init, Model):

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TIC(self.wm.ev, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class PVLDecayTDC(PVL_init, Model):

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TDC(self.wm.ev, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


class PVLDeltaTDC(PVL_init, Model):

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TDC(self.wm.ev, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


class EVLDecayTIC(EVL_init, Model):

    def __update__(self, choice, win, loss, trial):
        self.wm.utility = expect_utility(self.wm.win, self.wm.loss, self.par.W)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TIC(self.wm.ev, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class EVLDeltaTIC(EVL_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = expect_utility(self.wm.win, self.wm.loss, self.par.W)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TIC(self.wm.ev, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class EVLDecayTDC(EVL_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = expect_utility(self.wm.win, self.wm.loss, self.par.W)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TDC(self.wm.ev, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


class EVLDeltaTDC(EVL_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = expect_utility(self.wm.win, self.wm.loss, self.par.W)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.theta = TDC(self.wm.ev, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


class VPPDecayTIC(VPP_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        self.wm.epi = cal_epi(self.wm.payoff, self.par.eP, self.par.eN)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.P = Decay(self.wm.P, self.wm.epi, choice, self.par.k)
        self.wm.V = self.par.w * self.wm.ev + (1 - self.par.w) * self.wm.P
        self.wm.theta = TIC(self.wm.V, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class VPPDeltaTIC(VPP_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        self.wm.epi = cal_epi(self.wm.payoff, self.par.eP, self.par.eN)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.P = Decay(self.wm.P, self.wm.epi, choice, self.par.k)
        self.wm.V = self.par.w * self.wm.ev + (1 - self.par.w) * self.wm.P
        self.wm.theta = TIC(self.wm.V, self.par.cons)
        self.wm.Pr = convert_Pr(self.wm.theta)


class VPPDecayTDC(VPP_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        self.wm.epi = cal_epi(self.wm.payoff, self.par.eP, self.par.eN)
        [_, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, 1)
        self.wm.ev = Decay(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.P = Decay(self.wm.P, self.wm.epi, choice, self.par.k)
        self.wm.V = self.par.w * self.wm.ev + (1 - self.par.w) * self.wm.P
        self.wm.theta = TDC(self.wm.V, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


class VPPDeltaTDC(VPP_init, Model):
    def __update__(self, choice, win, loss, trial):
        self.wm.utility = prospect_utility(self.wm.payoff, self.par.alpha, 1, self.par.losslambda)
        self.wm.epi = cal_epi(self.wm.payoff, self.par.eP, self.par.eN)
        [self.wm.ev, self.wm.PE] = Delta(self.wm.ev, self.wm.utility, choice, self.par.A)
        self.wm.P = Decay(self.wm.P, self.wm.epi, choice, self.par.k)
        self.wm.V = self.par.w * self.wm.ev + (1 - self.par.w) * self.wm.P
        self.wm.theta = TDC(self.wm.V, self.par.cons, trial)
        self.wm.Pr = convert_Pr(self.wm.theta)


def get_model_from_str(model_name):
    return eval(model_name)


def get_MLE(model_name, par, win, loss, ch, all_trials=False):
    '''
    compute the log likelihood of one model for one subject
    '''
    T = len(ch)
    model = get_model_from_str(model_name)(par)
    MLE = 0
    if not all_trials:
        T -= 1
    for trial in range(T):
        choice = ch[trial] - 1
        model.update(choice, win[trial], loss[trial], trial + 1)  # matlab: trial from 1:'',python trial from 0
        MLE += model.wm.MLE1trial

    return MLE


def load_subject_data(group_name):
    mat = loadmat('OriData/%sdata' % group_name)
    ch = mat['ch'].astype(np.int32)  # subnum,trialnum
    win = mat['win'].astype(np.float64)
    loss = mat['loss'].astype(np.float64)
    subname = None
    if 'subname' in mat:
        subname = mat['subname'].astype(np.int32).squeeze()
    return ch, win, loss, subname


def load_fit_data(group_name, model_name):
    mat = loadmat('FitData/%s_%s_result' % (model_name, group_name))
    finalmle = mat['finalmle'].astype(np.float64).squeeze() # the best log likelihood of each subject
    finalpar = mat['finalpar'].astype(np.float64)# the best parameters of each subject
    allmle = mat['MAXMLE'].astype(np.float64)  # [subnum,run_num] log likelihood of each subject of each searching seed
    allpar = mat['MAXX'].astype(np.float64)  # [subnum,run_num,par_num] parameters of each subject of each searching seed
    run_num = allmle.shape[1]
    return finalmle, finalpar, allmle, allpar, run_num


def check_all_model():
    '''
    check that the fitted log likelihood results are accurate for all models and all subjects
    '''
    for group_name in all_group_name:
        ch, win, loss, _ = load_subject_data(group_name)
        for model_name in all_paper_model[:-1]:
            finalmle, finalpar, allmle, allpar, run_num = load_fit_data(group_name, model_name)
            print(group_name, model_name)
            for sub in range(len(finalmle)):
                MLE = get_MLE(model_name, finalpar[sub], win[sub], loss[sub], ch[sub])
                # print(sub, MLE)


if __name__ == '__main__':
    check_all_model()
