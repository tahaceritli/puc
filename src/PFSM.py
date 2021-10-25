from copy import deepcopy
import numpy as np
import numpy.ma as ma
import functools
from src.utils import log_sum_exp, edit_distance

def log_sum_probs(log_p1, log_p2):
    log_mx = np.max([log_p1, log_p2])

    return log_mx + np.log(np.exp(log_p1 - log_mx) + np.exp(log_p2 - log_mx))

def ma_multidot(arrays):
    return functools.reduce(ma.dot, arrays)

from greenery.lego import parse

LOG_EPS = -1e150
PRINT = False

class Machine(object):
    def __init__(self):
        self.states = []
        self.T = {}
        self.T_backup = {}
        self.alphabet = []

    def create_pfsm_from_fsm(self,):
        fsm_obj = parse(self.reg_exp).to_fsm()

        self.alphabet = list(set([str(i) for i in list(fsm_obj.alphabet)]) - set(['anything_else',]))

        states = list(fsm_obj.states)
        self.add_states(states)

        initials = [fsm_obj.initial,]
        I = [np.log(1 / len(initials)) if state in initials else LOG_EPS for state in self.states]
        self.set_I(I)
        self.I_backup = self.I.copy()

        finals = list(fsm_obj.finals)
        F = [np.log(self.STOP_P) if state in finals else LOG_EPS for state in self.states]
        self.set_F(F)
        self.F_backup = self.F.copy()

        transitions = fsm_obj.map
        for state_i in transitions:
            trans = transitions[state_i]

            for symbol in list(trans):
                if str(symbol) == 'anything_else':
                    del trans[symbol]
            transitions[state_i] = trans

        for state_i in transitions:
            trans = transitions[state_i]
            state_js = np.array(list(trans.values()))
            if len(state_js) == 0:
                self.F[state_i] = 0.
            else:
                symbols_js = np.array(list(trans.keys()))
                if self.F[state_i] != LOG_EPS:
                    probs = np.array([(1.0 - np.exp(self.F[state_i])) / len(symbols_js) for i in range(len(symbols_js))])
                else:
                    probs = np.array([1.0 / len(symbols_js) for i in range(len(symbols_js))])

                for state_j in np.unique(state_js):
                    idx = np.where(state_js == state_j)[0]
                    symbols = list(symbols_js[idx])
                    self.add_transitions(state_i, state_j, symbols, list(probs[idx]))

    def add_state(self, state_name):
        if state_name not in self.states:
            self.states.append(state_name)

    def add_states(self, state_names):
        for state_name in state_names:
            if state_name not in self.states:
                self.states.append(state_name)
                self.T[state_name] = {}
                self.T_backup[state_name] = {}

    def add_transition(self):
        pass

    def add_transitions(self, i, j, obs, probs):
        for obs, prob in zip(obs,probs):
            if obs not in self.T[i]:
                self.T[i][obs] = {}
                self.T_backup[i][obs] = {}
            self.T[i][obs][j] = np.log(prob)
            self.T_backup[i][obs][j] = np.log(prob)

            # for faster search later
            if obs not in self.alphabet:
                self.alphabet.append(obs)

    def set_I(self, _I):
        self.I = {state:i for state,i in zip(self.states, _I)}

    def set_F(self, _F):
        self.F = {state:f for state,f in zip(self.states, _F)}

    def create_T_new(self):
        T_new = {}
        for a in self.T:
            for b in self.T[a]:
                if b not in T_new:
                    T_new[b] = np.ones((len(self.states), len(self.states))) * LOG_EPS

                for c in self.T[a][b]:
                    T_new[b][self.states.index(a),self.states.index(c)] = self.T[a][b][c]

        self.T_new = T_new

    def count_number_params(self):
        num_params = 0
        for a in self.T:
            if self.I[a] != LOG_EPS or self.F[a] !=LOG_EPS:
                num_params += 1

            for b in self.T[a]:
                for c in self.T[a][b]:
                    num_params += 1
        return num_params

    def num_states(self):
        return len(self.states)

    def find_possible_targets(self, current_state, word, current_index, p):
        # repeat at a given state
        repeat_p = 0

        while current_state == self.repeat_state and self.repeat_count != 0:
            alpha = word[current_index]
            if alpha in self.T[current_state]:
                if current_state in self.T[current_state][alpha]:
                    repeat_p += self.T[current_state][alpha][current_state]
                    current_index += 1
                    self.repeat_count -= 1
                else:
                    self.candidate_path_prob = 0
                    self.ignore = True
                    break
            else:
                self.candidate_path_prob = 0
                self.ignore = True
                break

        if current_index == len(word):
            if self.F[current_state] != LOG_EPS:
                if self.candidate_path_prob == 0:
                    self.candidate_path_prob = p + self.F[current_state]
                else:
                    self.candidate_path_prob = log_sum_probs(self.candidate_path_prob, p + self.F[current_state])
        else:
            if not self.ignore:
                alpha = word[current_index]
                if PRINT:
                    print('\tcurrent_state', current_state)
                    print('\tchar =', alpha)
                if alpha in self.T[current_state]:
                    for target_state_name in self.T[current_state][alpha]:
                        tran_p = self.T[current_state][alpha][target_state_name]
                        self.find_possible_targets(target_state_name, word, current_index + 1, p + tran_p + repeat_p)

    def find_possible_targets_counts(self, current_state, word, current_index, p, q, _alpha, q_prime):
        # repeat at a given state
        repeat_p = 0

        while current_state == self.repeat_state and self.repeat_count != 0:
            alpha = word[current_index]
            if alpha in self.T[current_state]:
                if current_state in self.T[current_state][alpha]:
                    repeat_p += self.T[current_state][alpha][current_state]
                    if current_state == q and alpha == _alpha and current_state == q_prime:
                        self.candidate_path_parameter_count += 1
                    current_index += 1
                    self.repeat_count -= 1
                else:
                    self.candidate_path_prob = 0
                    self.candidate_path_parameter_count = 0
                    self.ignore = True
                    break
            else:
                self.candidate_path_prob = 0
                self.ignore = True
                break

        if current_index == len(word):
            if self.F[current_state] != LOG_EPS:
                if self.candidate_path_prob == 0:
                    self.candidate_path_prob = p + self.F[current_state]
                else:
                    self.candidate_path_prob = log_sum_probs(self.candidate_path_prob, p + self.F[current_state])
        else:
            if not self.ignore:
                alpha = word[current_index]
                if alpha in self.T[current_state]:
                    for target_state_name in self.T[current_state][alpha]:
                        tran_p = self.T[current_state][alpha][target_state_name]
                        if current_state == q and alpha == _alpha and target_state_name == q_prime:
                            self.candidate_path_parameter_count += 1
                        self.find_possible_targets_counts(target_state_name, word, current_index + 1, p + tran_p + repeat_p, q, _alpha, q_prime)

    def find_possible_targets_counts_final(self, current_state, word, current_index, p, final_state):
        # repeat at a given state
        repeat_p = 0

        while current_state == self.repeat_state and self.repeat_count != 0:
            alpha = word[current_index]
            if alpha in self.T[current_state]:
                if current_state in self.T[current_state][alpha]:
                    repeat_p += self.T[current_state][alpha][current_state]
                    current_index += 1
                    self.repeat_count -= 1
                else:
                    self.candidate_path_prob = 0
                    self.candidate_path_parameter_count = 0
                    self.ignore = True
                    break
            else:
                self.candidate_path_prob = 0
                self.ignore = True
                break

        if current_index == len(word):
            if self.F[current_state] != LOG_EPS:
                if self.candidate_path_prob == 0:
                    self.candidate_path_prob = p + self.F[current_state]
                else:
                    self.candidate_path_prob = log_sum_probs(self.candidate_path_prob, p + self.F[current_state])

                if current_state == final_state:
                    self.candidate_path_parameter_count = 1

        else:
            if not self.ignore:
                alpha = word[current_index]
                if alpha in self.T[current_state]:
                    for target_state_name in self.T[current_state][alpha]:
                        tran_p = self.T[current_state][alpha][target_state_name]
                        self.find_possible_targets_counts_final(target_state_name, word, current_index + 1, p + tran_p + repeat_p, final_state)

    def calculate_probability(self, word):
        if not self.supported_words[word]:
            return LOG_EPS
        else:
            # reset probability to 0
            self.word_prob = LOG_EPS

            # Find initial states with non-zero probabilities
            possible_init_states = []
            for state in self.states:
                if self.I[state] != LOG_EPS:
                    if len(word) > 0:
                        if word[0] in self.T[state]:
                            possible_init_states.append(state)
                    else:
                        possible_init_states.append(state)
            if PRINT:
                print('possible_init_states_names', possible_init_states)

            # Traverse each initial state which might lead to the given word
            for init_state in possible_init_states:
                self.ignore = False

                # reset path probability to 0
                self.candidate_path_prob = 0

                current_state = init_state
                if PRINT:
                    print('\tcurrent_state_name', current_state)

                self.find_possible_targets(current_state, word, 0, self.I[current_state])

                # add probability of each successful path that leads to the given word
                if self.candidate_path_prob !=0:
                    if self.word_prob == LOG_EPS:
                        self.word_prob = self.candidate_path_prob
                    else:
                        self.word_prob = log_sum_probs(self.word_prob, self.candidate_path_prob)

            return self.word_prob

    def forward_recursion(self, x):
        """
        :param x:
        :return: alpha_messages: alpha_messages[l] stores the message from l to l+1 where l in {0,...,L}
        """
        alpha_messages = []
        alpha_messages.append(np.exp(np.array(list(self.I.values()))))
        for l, alpha in enumerate(x[:-1]):
            if alpha not in self.T_new:
                alpha_messages.append(np.zeros(len(alpha_messages[l]))) #np.dot(alpha_messages[l], np.zeros(len(alpha_messages[l]))))
            else:
                alpha_messages.append(np.dot(alpha_messages[l], np.exp(self.T_new[alpha])))
                if np.max(alpha_messages[-1]) != 0.:
                    alpha_messages[-1] = alpha_messages[-1]/alpha_messages[-1].sum()

        return alpha_messages

    def backward_recursion(self, x):
        """
        :param x:
        :return: beta_messages : beta_messages[l] stores the message from l+1 to l where l in {0,...,L}
        """
        beta_messages = []
        beta_messages.append(np.exp(np.array(list(self.F.values()))))
        for l, alpha in enumerate(reversed(x[1:])):
            if alpha not in self.T_new:
                beta_messages = [np.zeros(len(beta_messages[0]))] + beta_messages
            else:
                beta_messages = [np.dot(np.exp(self.T_new[alpha]), beta_messages[0])] + beta_messages
                if np.max(beta_messages[0]) != 0.:
                    beta_messages[0] = beta_messages[0]/beta_messages[0].sum()

        return beta_messages

    def run_forward_backward(self, x):
        self.alpha_messages = self.forward_recursion(x)
        self.beta_messages = self.backward_recursion(x)
        joint_probs = []
        for l in range(len(x)):
            joint_probs.append(self.calculate_derivative_temp(l, x))

        return joint_probs

    def calculate_derivative_temp(self, l, x):
        # l is in 0...L

        if x[l] not in self.T_new:
            smoothing_probs = np.zeros((len(self.alpha_messages[0]), len(self.alpha_messages[0])))
        else:
            smoothing_probs = np.outer(self.alpha_messages[l], self.beta_messages[l]) * np.exp(self.T_new[x[l]])

        if np.max(smoothing_probs) != 0.:
            smoothing_probs = smoothing_probs / smoothing_probs.sum()

        return smoothing_probs

    def calculate_probability_new(self, word):

        if not self.supported_words[word]:
            return LOG_EPS
        else:
            temp = np.array(list(self.I.values()))
            temp2 = np.array(list(self.F.values()))

            temp_T = []
            for w in word:
                if w in self.T_new:
                    temp_T = temp_T + [ma.masked_where(self.T_new[w] == LOG_EPS, self.T_new[w])]
                else:
                    return LOG_EPS

            Xs = [ma.masked_where(temp == LOG_EPS, temp)] + temp_T + [ma.masked_where(temp2 == LOG_EPS, temp2)]
            max_Xs = [np.max(X) for X in Xs]
            exp_Xs = [np.exp(X - max_X) for X, max_X in zip(Xs, max_Xs)]
            res = ma_multidot(exp_Xs)
            res_prob = np.ma.getdata(res)
            if res_prob == 0 or res.mask:
                return LOG_EPS
            else:
                return np.log(res_prob) + sum(max_Xs)

    def count_c_final(self, current_state, final_state, word, current_index, count, log_sm):
        TRAIN_PRINT = False
        if current_index == len(word):
            if self.F[current_state] != LOG_EPS:
                if current_state == final_state:
                    self.candidate_path_C = log_sm
                if TRAIN_PRINT:
                    print('count =', count)
                    print('log_sm =,', log_sm)
                    print('final candidiate_path_C = ', self.candidate_path_C)
                    print('self.T[self.a][self.b][self.c]) = ', self.T[self.a][self.b][self.c])
        else:
            x_i_n = word[current_index]
            if x_i_n in self.T[current_state]:
                for target_state_name in self.T[current_state][x_i_n]:
                    self.count_c_final(target_state_name, final_state, word, current_index + 1, count, log_sm + self.T[current_state][x_i_n][target_state_name])

    def calculate_gradient_abc_new_optimized(self, word, q, alpha, q_prime):
        # Find initial states with non-zero probabilities
        if len(word) == 0:
            return 0
        else:
            possible_init_states = []
            for state in self.states:
                if self.I[state] != LOG_EPS:
                    possible_init_states.append(state)

            # Traverse each initial state which might lead to the given word
            for init_state in possible_init_states:
                self.ignore = False

                # reset path probability to 0
                self.candidate_path_prob = 0
                self.candidate_path_parameter_count = 0

                if self.repeat_state is not None:
                    self.repeat_count = 4
                self.find_possible_targets_counts(init_state, word, 0, self.I[init_state], q, alpha, q_prime)

                # break when a successful path is found, assuming there'll only be one successful path. check if that's the case.
                if self.candidate_path_parameter_count != 0:
                    break

            return self.candidate_path_parameter_count

    def calculate_gradient_abc_new_optimized_marginals(self, marginals, word, q, alpha, q_prime):
        # Find initial states with non-zero probabilities
        if len(word) == 0:
            return 0
        else:
            indices = np.where(list(word) == alpha)[0]
            return sum([marginals[ind][self.states.index(q), self.states.index(q_prime)] for ind in indices])

    def calculate_gradient_initial_state_optimized(self, x_i, initial_state):
        if len(x_i) == 0:
            return 0
        else:
            return int(x_i[0] in self.T[initial_state])

    def calculate_gradient_final_state(self, x_i, initial_state, final_state):
        TRAIN_PRINT = False
        # reset gradient to 0
        gradient = 0.

        # Traverse each initial state which might lead to the given word with the given final state
        self.candidate_path_C = 0.
        log_mx = self.I[initial_state]
        # if self.F[final_state] != LOG_EPS:
        #     log_mx += self.F[final_state]
        self.count_c_final(initial_state, final_state, x_i, 0, 0, log_mx)

        if self.candidate_path_C != 0:
            if TRAIN_PRINT:
                print('self.candidate_path_C =', self.candidate_path_C)
            gradient += np.exp(self.candidate_path_C)

        # Multiply with other terms
        if TRAIN_PRINT:
            print('self.candidate_path_C, gradient =', self.candidate_path_C, gradient)

        return gradient

    def calculate_gradient_final_state_optimized(self, x_i, final_state):
        # Find initial states with non-zero probabilities
        if len(x_i) == 0:
            return 0
        else:
            possible_init_states = []
            for state in self.states:
                if self.I[state] != LOG_EPS:
                    possible_init_states.append(state)

            # Traverse each initial state which might lead to the given word
            for init_state in possible_init_states:
                self.ignore = False

                # reset path probability to 0
                self.candidate_path_prob = 0
                self.candidate_path_parameter_count = 0

                if self.repeat_state is not None:
                    self.repeat_count = 4

                self.find_possible_targets_counts_final(init_state, x_i, 0, self.I[init_state], final_state)

                # break when a successful path is found, assuming there'll only be one successful path. check if that's the case.
                if self.candidate_path_parameter_count != 0:
                    break

            return self.candidate_path_parameter_count

    def copy_to_z(self):
        self.I_z = deepcopy(self.I)
        self.T_z = deepcopy(self.T)
        self.F_z = deepcopy(self.F)

################################# MACHINES ##################################
############# MISSINGS #################
class MissingsNew(Machine):
    def __init__(self):
        self.alphabet = ['NULL', 'null', 'Null', 'NA', 'NA ', ' NA', 'N A', 'N/A', 'N/ A', 'N /A', 'N/A', '#NA', '#N/A',
                         'na', ' na', 'na ', 'n a', 'n/a', 'N/O', 'NAN', 'NaN', 'nan', '-NaN', '-nan', '-', '!', '?',
                         '*', '.', '0', '-1', '-9', '-99', '-999', '-9999', '-99999', '', ' ']
        self.LEN_1_PROB = 1e-7
        self.set_probs()

    def set_probs(self):
        self.probs = {alpha: np.log(self.LEN_1_PROB) if len(alpha) == 1 else np.log((1.0 - self.LEN_1_PROB) / (len(self.alphabet) - 7)) for alpha in self.alphabet}

    def calculate_probability(self, word):
        self.ignore = False
        if word in self.alphabet:
            return self.probs[word]
        else:
            return LOG_EPS

    def calculate_probability_new(self, word):
        self.ignore = False
        if word in self.alphabet:
            return self.probs[word]
        else:
            return LOG_EPS

class MissingsNew(Machine):
    def __init__(self):
        self.alphabet = ['NULL', 'null', 'Null', 'NA', 'NA ', ' NA', 'N A', 'N/A', 'N/ A', 'N /A', 'N/A', '#NA', '#N/A',
                         'na', ' na', 'na ', 'n a', 'n/a', 'N/O', 'NAN', 'NaN', 'nan', '-NaN', '-nan', '-', '!', '?',
                         '*', '.', '0', '-1', '-9', '-99', '-999', '-9999', '-99999', '', ' ']
        self.LEN_1_PROB = 1e-7
        self.set_probs()

    def set_probs(self):
        self.probs = {alpha: np.log(self.LEN_1_PROB) if len(alpha) == 1 else np.log((1.0 - self.LEN_1_PROB) / (len(self.alphabet) - 7)) for alpha in self.alphabet}

    def calculate_probability(self, word):
        self.ignore = False
        if word in self.alphabet:
            return self.probs[word]
        else:
            return LOG_EPS

    def calculate_probability_new(self, word):
        self.ignore = False
        if word in self.alphabet:
            return self.probs[word]
        else:
            return LOG_EPS

class AnomalyNew(Machine):
    def __init__(self):
        self.states = []
        self.alphabet = [chr(i) for i in range(1114112)]
        self.STOP_P = 1e-14
        self.T = {}
        self.T_backup = {}
        self.add_states(['q_unknown', 'q_unknown_3'])
        self.set_I([np.log(1.) if state == 'q_unknown' else LOG_EPS for state in self.states])
        self.I_backup = self.I.copy()
        self.set_F([np.log(self.STOP_P) if state == 'q_unknown_3' else LOG_EPS for state in self.states])
        self.F_backup = self.F.copy()

    def calculate_probability(self, word):
        self.ignore = False
        if self.supported_words[word] and len(word)!=0:
            if len(word) > 100:
                return np.log((1.-self.STOP_P)/len(self.alphabet)) * 100 + np.log(self.STOP_P)
            else:
                return np.log((1.-self.STOP_P)/len(self.alphabet)) * len(word) + np.log(self.STOP_P)
        else:
            return LOG_EPS

    def calculate_probability_new(self, word):
        self.ignore = False
        if self.supported_words[word] and len(word) != 0:
            if len(word) > 100:
                return np.log((1. - self.STOP_P) / len(self.alphabet)) * 100 + np.log(self.STOP_P)
            else:
                return np.log((1. - self.STOP_P) / len(self.alphabet)) * len(word) + np.log(self.STOP_P)
        else:
            return LOG_EPS

# I'm probably not going to use this.
class AnomalyUnit(Machine):
    def __init__(self):
        self.states = []
        self.alphabet = [chr(i) for i in range(1114112)]
        self.STOP_P = 1e-14
        self.T = {}
        self.T_backup = {}
        self.add_states(['q_unknown', 'q_unknown_3'])
        self.set_I([np.log(1.) if state == 'q_unknown' else LOG_EPS for state in self.states])
        self.I_backup = self.I.copy()
        self.set_F([np.log(self.STOP_P) if state == 'q_unknown_3' else LOG_EPS for state in self.states])
        self.F_backup = self.F.copy()

    def calculate_probability(self, word, symbols, t):
        gamma = 2
        u_t = symbols[t]
        x = [- gamma * edit_distance(word, u_tl) for u_tl in u_t]

        return log_sum_exp(x)


class FloatsNewAuto(Machine):
    def __init__(self, ):
        self.STOP_P = 4 * 1e-5
        self.states = []
        self.T = {}
        self.T_backup = {}
        self.alphabet = []
        self.repeat_count = 0
        self.repeat_state = None
        self.reg_exp = "[\-+]?(((\d+(\.\d*)?)|\.\d+)([eE][\-+]?[0-9]+)?)|(\d{1,3}(,[0-9]{3})+(\.\d*)?)"
        self.create_pfsm_from_fsm()
        self.create_T_new()
        self.copy_to_z()

    def calculate_probability(self, word):

        if (not self.supported_words[word]) or word == '.':
            return LOG_EPS
        else:

            # reset probability to 0
            self.word_prob = LOG_EPS

            # Find initial states with non-zero probabilities
            possible_init_states = []
            for state in self.states:
                if self.I[state] != LOG_EPS:
                    if len(word) > 0:
                        if word[0] in self.T[state]:
                            possible_init_states.append(state)
                    else:
                        possible_init_states.append(state)
            if PRINT:
                print('possible_init_states_names', [temp.name for temp in possible_init_states])

            # Traverse each initial state which might lead to the given word
            for init_state in possible_init_states:
                self.ignore = False
                # reset path probability to 0
                self.candidate_path_prob = 0

                current_state = init_state
                if PRINT:
                    print('\tcurrent_state_name', current_state.name)

                self.find_possible_targets(current_state, word, 0, self.I[current_state])

                # add probability of each successful path that leads to the given word
                if self.candidate_path_prob != 0:
                    if self.word_prob == LOG_EPS:
                        self.word_prob = self.candidate_path_prob
                    else:
                        self.word_prob = log_sum_probs(self.word_prob, self.candidate_path_prob)

            return self.word_prob