from OpenNeural import *


# GNU AFFERO GPL (c) Useop Gim 2022
# Please let reference
# If you intereseted in my work visit in
# https://sites.google.com/view/gesope/projects/a-i/neural-network-algorithm-explanation?authuser=0

# basic shuffle random
def REPLAY_SHUFFLE(replay_buffer) -> np.ndarray:
    index_list = np.arange(len(replay_buffer))
    np.random.shuffle(index_list)
    return index_list


# prioritization
def REPLAY_PRIORITIZATION(rl_model, neural_networks, replay_buffer) -> np.ndarray:
    robust_priority = np.empty(0)
    for exp in replay_buffer:
        s, a, r, _s, t = exp
        err = rl_model([s, a, r, _s, t], 0.8, neural_networks[0], neural_networks[1])[-1]
        robust_priority = np.append(robust_priority, err)
    ordered_index = np.argsort(robust_priority)[::-1]  # ordered index by error distance
    prob_replay = 1 / np.arange(1, len(replay_buffer) + 1, 1)  # robust probability
    annealing_bias = (1 / len(replay_buffer) * prob_replay) ** 0.5  # robust probability and annealing the bias
    prob_random = np.random.uniform(0, 1, len(replay_buffer))  # random probability list
    prob_picked = np.argsort(np.where(prob_random < annealing_bias, 0, 1))  # sorting picked probabilities
    return ordered_index[prob_picked]  # ordered index by sorting picked probabilities


# slice the replay buffer by replay size [return call by object reference]
def RL_REPLAY_SLICE(replay_buffer, replay_trial, replay_sz):
    mini_batch = replay_buffer[replay_trial * replay_sz:(replay_trial + 1) * replay_sz]
    return mini_batch


# learning from mini batch array [call by object reference ( c pointer array)] (for latter C work)
def RL_MINI_ACCUMU_GRAD(rl_model, neural_networks, mini_batch, replay_trial) -> list:
    """
    This function will be start when the size of replay_buffer reaches to the replay_max_sz

    Args:
        rl_model(function) : the rl algorithm
        neural_networks(list): openNeural list [main and target]
        mini_batch(np.ndarray) : the mini batch buffer
    Return:
         neural_networks(openNeural)
    """
    assert 0 <= replay_trial < len(mini_batch)
    accumulate_grad_q = np.zeros(neural_networks[0].get_shape()[-1])
    accumulate_grad_yt = np.zeros(neural_networks[1].get_shape()[-1])
    for mini_exp in mini_batch[replay_trial]:
        if len(mini_exp) == 5:
            V, Q, E = rl_model(mini_exp, 0.8, neural_networks[0], neural_networks[1])
            accumulate_grad_yt += V
            accumulate_grad_q += Q
        else:
            break
    return [accumulate_grad_q, accumulate_grad_yt]


# update target function  [call by object reference ( c pointer array)] (for latter C work)
def RL_TRG_UPDATE(t_update_step, t_update_interval, t_update_rate, qn, qn_target):
    """
    Args:
        t_update_step(int): update iteration
        t_update_interval(int): update interval
        t_update_rate(float): soft update rate
        qn(openNeural): main neural network
        qn_target(openNeural): target neural network
    """
    assert 0 < t_update_rate < 1
    if t_update_step % t_update_interval == 0:
        if t_update_rate != 1:
            qn_target.set_w_layer(qn.get_layer()[0] * t_update_rate \
                                  + qn_target.get_layer()[0] * (1.0 - t_update_rate))
            qn_target.set_b_layer(qn.get_layer()[1] * t_update_rate \
                                  + qn_target.get_layer()[1] * (1.0 - t_update_rate))
        else:
            qn_target.set_w_layer(qn.get_layer()[0])
            qn_target.set_b_layer(qn.get_layer()[1])


def DQN(exp, gamma, qn, tqn) -> list:
    """
        Args:
            exp(np.ndarray): a list [s,a,r,s',t]
            gamma(float): a discount rate
            qn(openNeural) : Q value network
            tqn(openNeural) : Target Q value network
        Returns:
            yt, q, error
    """
    assert len(exp) == 5
    assert 0 < gamma <= 1
    s, a, r, _s, t = exp
    yt = qn.run(s)
    if not t:
        yt[a] = r + gamma * max(tqn.run(_s))
    else:
        yt[a] = r
    return [yt, qn.run(s), 0.5 * (yt[a] - qn.output[a])]


def DDQN(exp, gamma, qn, tqn) -> list:
    """
        Args:
            exp(np.ndarray): a list [s,a,r,s',t]
            gamma(float): a discount rate
            qn(openNeural) : Q value network
            tqn(openNeural) : Target Q value network
        Returns:
            yt, q, error
    """
    assert len(exp) == 5
    assert 0 < gamma <= 1
    s, a, r, _s, t = exp
    yt = qn.run(s)
    if not t:
        yt[a] = r + gamma * tqn.run(_s)[np.argmax(qn.run(_s))]
    else:
        yt[a] = r
    return [yt, qn.run(s), 0.5 * (yt[a] - qn.run(s)[a])]


def D2QN(exp, gamma, qn, tqn) -> list:
    """
        Args:
            exp(np.ndarray): a list [s,a,r,s',t]
            gamma(float): a discount rate
            qn(openNeural) : Q value network
            tqn(openNeural) : Target Q value network
        Returns:
            yt, q, error
    """
    assert len(exp) == 5
    assert 0 < gamma <= 1
    s, a, r, _s, t = exp
    qn.run(s)
    q = (qn.output[-1]  # V
         + qn.output  # A
         - np.mean(qn.output[:-1]))  # mean A
    yt = q.copy()
    if not t:
        tqn.run(_s)
        yt[a] = r + gamma * (tqn.output[-1]  # V
                             + tqn.output[:-1][a]  # A
                             - np.mean(tqn.output[:-1]))  # mean A
    else:
        yt[a] = r
    return [yt, q, 0.5 * (yt[a] - qn.output[a])]


def D3QN(exp, gamma, qn, tqn) -> list:
    """
        Args:
            exp(np.ndarray): a list [s,a,r,s',t]
            gamma(float): a discount rate
            qn(openNeural) : Q value network
            tqn(openNeural) : Target Q value network
        Returns:
            yt, q, error
    """
    assert len(exp) == 5
    assert 0 < gamma <= 1
    s, a, r, _s, t = exp
    qn.run(s)
    q = (qn.output[-1]  # V
         + qn.output  # A
         - np.mean(qn.output[:-1]))  # mean A
    _a = np.argmax(qn.run(_s)[:-1])
    yt = q.copy()
    if not t:
        tqn.run(_s)
        yt[a] = r + gamma * (tqn.output[-1]  # V
                             + tqn.output[:-1][_a]  # A
                             - np.mean(tqn.output[:-1]))  # mean A
    else:
        yt[a] = r
    return [yt, q, 0.5 * (yt[a] - qn.run(s)[a])]


def RL_ACTION(s, agent, act_sz, epsilon) -> int:
    """
    Args:
        s(np.ndarray) : current state
        agent(openNeural) : neural class
        act_sz(int) : action size
        epsilon(float) : between 0 and 1 portability
    Return:
        action(int): action from fn(s)
    """
    if 0 < epsilon and random.uniform(0, 1) < epsilon:
        return random.randint(0, act_sz - 1)
    return int(np.argmax(agent.run(s)[:act_sz]))


# later works
def SAC(exp, alpha_gamma, QN, TQN) -> object:
    """
        Args:
            QN(list): [qn, qn_2, pn]
            TQN(list): [tqn, tqn_2]
            alpha_gamma(list): [alpha, gamma]
        """
    s, a, r, _s, t = exp
    alpha, gamma = alpha_gamma
    qn, qn_2, pn = QN[0], QN[1], QN[2]
    tqn, tqn_2 = TQN[0], TQN[1]
    q_list = np.array([qn.run(np.append(s, pn.run(s))),
                       qn_2.run(np.append(s, pn.run(s)))])
    tq_list = np.array([tqn.run(np.append(_s, pn.run(s))),
                        tqn_2.run(np.append(_s, pn.run(s)))])
    candidate = np.array([JSD(pn.run(s), softmax(q_list[0])), JSD(pn.run(s), softmax(q_list[1]))])
    index = np.argmax(candidate)
    # ---------------------- Update Q value ------------------------------------------------
    Q_s_a = q_list[index]
    # V_s_1 = Q'(s_1,a) - a*log(pi(s)) # Update for Critic Network
    # Jq = Q(s,a) - (r + g * V_s_1)
    V_s_1 = (tq_list[index] - alpha * np.log1p(pn.run(_s)))
    T_s_1 = (r + gamma * V_s_1)
    # ---------------------- Update policy -------------------------------------------------
    # Jp = -Q(s,a) + a*log(p(s))
    P_l = alpha * pn.run(s)
    # Update alpha function
    alpha = min(max(max(- pn.run(s) + 0.005), 0.0001), 2)
    return Q_s_a, T_s_1, \
           P_l, Q_s_a, \
           index, shannon_entropy(P_l)


class openRL:
    """
    This is DQN class.
    After declare, following below process\n
    RL_DQN_SETTING : Make the neural network
    """
    __environment = object
    __reward_system = object
    __RL_MODEL = object

    __alpha = 0.5  # policy temperature parameter
    __gamma = 0.8  # discount rate
    __tau = 0.001  # soft update rate
    __t_update_interval = 10
    __up_step = 0
    __it = 0
    __ep = 0
    __max_iter = 0
    __max_epoch = 0
    __epsilon = 0
    __number = 0
    reward_time_stamp = np.array(0)

    qn = openNeural()  # value neural
    tqn = openNeural()  # target value neural

    __replay_buffer = np.ndarray  # structure as [state, action, reward, future, termination, TDerror]
    __replay_trial = int
    __replay_buffer_max_sz = int
    __replay_sz = int
    __replay_fn = object
    __w_update_interval = 4
    __replay_index_list = np.empty((0, 6))
    __p_sum = 0
    __isw_list = np.ndarray

    def __init__(self, model, random_seed):
        """
        It initializes tje target and main neural network
        """
        self.tqn = openNeural()
        self.qn = openNeural()
        self.__p_sum = 0
        self.__random_seed = random_seed
        self.__RL_MODEL = model
        np.random.seed(seed=random_seed)  # for testing
        random.seed(random_seed)

    def GET_NEURON(self):
        return self.qn

    def RL_ADD_EXP(self, exp, replay_buffer, replay_buffer_max_sz) -> np.ndarray:
        """
        Args:
            exp(np.ndarray(5, dtype = object)): ['state', 'action', 'reward', 'future state', 'termination']
            replay_buffer(np.ndarray) : replay_buffer
            replay_buffer_max_sz(int) : replay_buffer maximum size
        """
        if len(replay_buffer) + 1 > replay_buffer_max_sz:
            replay_buffer = np.delete(replay_buffer, 0, axis=0)
        s, a, r, _s, t = exp
        replay_buffer = np.vstack((replay_buffer,
                                   np.array([s, a, r, _s, t], dtype=object)))
        return replay_buffer

    def RL_REPLAY(self, update_step):
        if len(self.__replay_buffer) == self.__replay_buffer_max_sz and update_step % self.__w_update_interval == 0:
            # ----------ordering replay----------
            self.__replay_index_list = REPLAY_PRIORITIZATION(
                self.__RL_MODEL, [self.qn, self.tqn], self.__replay_buffer)
            re_ordered_replay = self.__replay_buffer[self.__replay_index_list]
            # ---------mini batch replay---------
            mini_batch = np.array_split(re_ordered_replay, self.__replay_buffer_max_sz / self.__replay_sz)
            aQ, aY = RL_MINI_ACCUMU_GRAD(self.__RL_MODEL, [self.qn, self.tqn], mini_batch, self.__replay_trial)
            self.qn.learn_start(aQ, aY)

    def RL_PROCESS_MERGE(self, initial_state, show=True) -> bool:
        """
        Args:
            initial_state(np.ndarray) : initial state
            show(bool): show each reward
        Require:
            Initialized and set Neural networks
            __RL_MODEL
        Return:
            False : playing terminated, True :playing continue
        """
        self.__update = 0
        # ----------playing episode----------
        max_reward = 0
        total_reward = 0
        action_probability = 0.8
        reward_threshold = 1
        self.qn.opt_reset()
        self.tqn.opt_reset()
        self.__replay_buffer = np.empty((0, 5))
        self.__up_step = 0
        for self.__ep in range(self.__max_epoch):
            # ----------playing----------
            s = initial_state.copy()  # initialize state S
            max_reward = max(total_reward, max_reward)
            total_reward = 0
            for self.__it in range(self.__max_iter):  # increase the interation
                self.__up_step += 1
                # ----------Get exp from action----------
                r, t = self.__reward_system(s)  # get [r, t] (reward and terminated state)
                a = RL_ACTION(s=s, agent=self.qn, act_sz=3, epsilon=action_probability)  # get a (action)
                ss = self.__environment(s, a)  # get ss (new state)
                exp = np.array([s, a, r, ss, t], dtype=object)
                # ----------Add experience buffer----------
                self.__replay_buffer = self.RL_ADD_EXP(exp, self.__replay_buffer, self.__replay_buffer_max_sz)
                # ----------replay learning------
                self.RL_REPLAY(self.__up_step)
                # ------target uupdate--------------------
                RL_TRG_UPDATE(self.__up_step, self.__t_update_interval, self.__tau, self.qn, self.tqn)
                # ---------e-greedy policy -----------------
                total_reward += r
                # ---------Info -----------------
                if total_reward >= reward_threshold and show:
                    np.set_printoptions(precision=3, suppress=True)
                    self.qn.run(self.__replay_buffer[-2][0])
                    print(
                        Fore.LIGHTBLUE_EX,
                        f'previo',
                        f's : {str(self.__replay_buffer[-2][0])}, a : {self.__replay_buffer[-2][1]}, r : {self.__replay_buffer[-2][2]}, s+1 : {str(self.__replay_buffer[-2][3])}, t : {self.__replay_buffer[-2][4]}',
                        f'q : {self.qn.run(s)}',
                        f'e : {self.__ep}/{self.__max_epoch}',
                        f'step {self.__up_step}',
                        f'update {self.__up_step % self.__w_update_interval == 0, self.__up_step % self.__t_update_interval == 0}',
                        f'i : {self.__it}/{self.__max_iter}',
                        f'b : {len(self.__replay_buffer)}',
                        f'epsilon : {action_probability:.2f}',
                        f'reward_threshold R {reward_threshold}',
                        f'total R : {total_reward}')
                    self.qn.run(self.__replay_buffer[-1][0])
                    print(
                        Fore.LIGHTYELLOW_EX,
                        f'result',
                        f's : {str(self.__replay_buffer[-1][0])}, a : {self.__replay_buffer[-1][1]}, r : {self.__replay_buffer[-1][2]}, s+1 : {str(self.__replay_buffer[-1][3])}, t : {self.__replay_buffer[-1][4]}',
                        f'q : {self.qn.run(s)}',
                        f'e : {self.__ep}/{self.__max_epoch}',
                        f'step {self.__up_step}',
                        f'update {self.__up_step % self.__w_update_interval == 0, self.__up_step % self.__t_update_interval == 0}',
                        f'i : {self.__it}/{self.__max_iter}',
                        f'b : {len(self.__replay_buffer)}',
                        f'epsilon : {action_probability:.2f}',
                        f'reward_threshold R {reward_threshold}',
                        f'total R : {total_reward}')
                    action_probability *= 0.9
                    reward_threshold += 1
                    action_probability = max(action_probability, 0.01)
                # ---------Termination current episode -----------------
                if t: break
                else: s = ss.copy()  # update new state
            self.reward_time_stamp = np.append(self.reward_time_stamp, total_reward)
            if total_reward >= 12:
                break
        return False

    def __hindsight_replay(self, replay_buffer) -> np.ndarray:
        """
        Args:
            replay_buffer(np.ndarray) : replay_buffer
        :return: added experiance in replay_buffer
        """
        __last = replay_buffer[-1]  # get last
        __goal = __last[3]  # set new goal
        for trajectory in replay_buffer:
            __s, __a, __r, __ss, __t = trajectory  # the replay buffer
            __r, __t = self.__reward_system(__s, __goal)  # change reward
            replay_buffer = self.RL_ADD_EXP(np.array([__s, __a, __r, __ss, __t], replay_buffer, dtype=object))
        return replay_buffer

    def RL_LEARN_SETTING(self,
                         enviro_fn=None, reward_fn=None, max_iter=60, max_epoch=1000,
                         buffer_maximum_sz=32, buffer_replay_sz=1, buffer_replay_trial=1,
                         dropout_rate=0.0,
                         learning_rate=0.005, learn_optima='NADAM', loss_fun=HUBER,
                         w_update_interval=5, t_update_interval=10, gamma=0.99, tau=0.01
                         ) -> None:
        """
        Args:
            enviro_fn(function):        The function input (s,a) then return [s']
            reward_fn(function):        The function input (s') then return [r,t]
            max_iter(int):              The # of maximum iteration(step) for one epoch
            max_epoch(int):             The # of maximum epoch for terminating RL
            buffer_maximum_sz(int):     The maximum # of experience buffer
            buffer_replay_sz(int):      The # of replay buffer size
            buffer_replay_trial(int):   The # of replaying
            replay_type(str):           The replay optimization
            dropout_rate(float):        The drop out for learning
            learning_rate(float):       The learning rate for learning
            loss_fun(function):         The loss(cost) function type for learning
            learn_optima(str):          The learning optimization function
            t_update_interval(int):                 The C step (update target) frequency
            gamma(float):               The discount variable between greater than 0 and 1
            tau(float):                 The update percentage between greater than 0 and 1
        """
        assert learn_optima == 'ADAM' or 'NADAM' or 'NONE'
        assert 0 < max_iter
        assert 0 < max_epoch
        assert 0 < buffer_maximum_sz
        assert 0 < buffer_replay_sz
        assert 0 < buffer_replay_trial
        assert buffer_replay_sz <= buffer_maximum_sz
        assert 0 < t_update_interval and str(type(t_update_interval)) == "<class 'int'>"
        assert 0.0 < gamma <= 1.0
        assert 0.0 < tau <= 1.0
        assert 0.0 < learning_rate
        assert 0.0 <= dropout_rate <= 1
        assert str(type(loss_fun)) == "<class 'function'>"

        self.qn = openNeural()
        self.tqn = openNeural()

        self.__environment = enviro_fn
        self.__reward_system = reward_fn
        self.__gamma = gamma
        self.__replay_buffer = np.empty((0, 6))
        self.__max_iter = max_iter
        self.__max_epoch = max_epoch
        self.__replay_buffer_max_sz = buffer_maximum_sz
        self.__replay_sz = buffer_replay_sz
        self.__replay_trial = buffer_replay_trial
        self.__w_update_interval = w_update_interval
        self.__tau = tau
        self.__t_update_interval = t_update_interval

        self.qn.opt_reset()
        self.qn.learning_set(
            gradient_clipping_norm=0,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_fun=loss_fun,
            learn_optima=learn_optima,
            processor='cpu')
        self.tqn.opt_reset()
        self.tqn.learning_set(
            gradient_clipping_norm=0,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            loss_fun=loss_fun,
            learn_optima=learn_optima,
            processor='cpu')

    def RL_SETTING(self,
                   q_layer=np.array([4, 8, 12, 3]),
                   q_activation_fn=
                   np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                   q_normalization=
                   np.array([linear_x, znormal, znormal, linear_x], dtype=object)
                   ) -> None:
        for i, L in enumerate(q_layer):
            self.qn.add_layer(L, q_activation_fn[i], normal=q_normalization[i])
        self.qn.generate_weight()
        self.qn.he_initialization()
        print('Q NEURAL : ', self.qn.get_shape())
        self.tqn << self.qn
        print('TARGET NEURAL : ', self.tqn.get_shape())

    def RL_SAVE(self, file_name) -> None:
        self.qn.numpy_save(file_name + "qn")
        self.tqn.numpy_save(file_name + "tqn")
        print("SAVE FINISH")

    def RL_LOAD(self, file_name) -> None:
        self.tqn.numpy_load(file_name + "tqn")
        self.qn.numpy_load(file_name + "qn")
        print("LOAD FINISH")


class RL_ON_POLICY(openRL):
    alpha = 0.5
    pn = openNeural
    qn_2 = openNeural
    tqn_2 = openNeural

    def RL_LEARNING_SAC(self, neural_networks, replay_exp, gamma) -> object:
        """
        Args:
            neural_networks(list): [qn_1, qn_2, pn, qn_1_target, qn_2_target]
            replay_exp(np.ndarray):
            gamma(float):
        """
        s, a, r, _s, t = replay_exp
        tq, q, tp, p, index, e = self.__RL_MODEL(s, a, r, _s, t, gamma, neural_networks[0:3], neural_networks[3:5])
        # ----------double Q learning----------
        qn_1, qn_2, pn, qn_1_target, qn_2_target = neural_networks  # call by reference for the future c code
        if index == 0:
            qn_1.learn_start(out_val=q, target_val=tq)
        else:
            qn_2.learn_start(out_val=q, target_val=tq)
        pn.learn_start(out_val=p, target_val=tp)
        return qn_1, qn_1_target, qn_2, qn_2_target, pn

    def RL_TRG_UPDATE(self, t_update_step, t_update_interval, t_update_rate, QN, TQN) -> object:
        """
        Args:
            t_update_step(int): update iteration
            t_update_interval(int): update interval
            t_update_rate(float): soft update rate
            qn(openNeural): main neural network
            qn_target(openNeural): target neural network
        """
        assert 0 < t_update_rate < 1
        qn_1, qn_2, pn = QN[0], QN[1], QN[2]
        tqn_1, tqn_2, index = TQN[0], TQN[1], TQN[2]
        if t_update_step % t_update_interval == 0:
            if t_update_rate != 1:
                if index == 0:
                    tqn_1.set_w_layer(qn_1.get_layer()[0] * self.__tau \
                                      + tqn_1.get_layer()[0] * (1.0 - self.__tau))
                    tqn_1.set_b_layer(qn_1.get_layer()[1] * self.__tau \
                                      + tqn_1.get_layer()[1] * (1.0 - self.__tau))
                else:
                    tqn_2.set_w_layer(qn_2.get_layer()[0] * self.__tau \
                                      + tqn_2.get_layer()[0] * (1.0 - self.__tau))
                    tqn_2.set_b_layer(qn_2.get_layer()[1] * self.__tau \
                                      + tqn_2.get_layer()[1] * (1.0 - self.__tau))
            else:
                if index == 0:
                    tqn_1.set_w_layer(qn_1.get_layer()[0])
                    tqn_1.set_b_layer(qn_1.get_layer()[1])
                else:
                    tqn_2.set_w_layer(qn_2.get_layer()[0])
                    tqn_2.set_b_layer(qn_2.get_layer()[1])
        return tqn_1, tqn_2

    def RL_SAVE(self, file_name) -> None:
        self.tqn.csv_save(file_name + "tqn")
        self.qn.csv_save(file_name + "qn")
        self.tqn_2.csv_save(file_name + "tqn_2")
        self.qn_2.csv_save(file_name + "qn_2")
        self.pn.csv_save(file_name + "pn")

    def RL_LOAD(self, file_name) -> None:
        self.tqn.csv_load(file_name + "tqn")
        self.qn.csv_load(file_name + "qn")
        self.tqn_2.csv_load(file_name + "tqn_2")
        self.qn_2.csv_load(file_name + "qn_2")
        self.pn.csv_load(file_name + "pn")


