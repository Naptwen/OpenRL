import numpy as np

from openNeural import *
import math

# GNU AFFERO GPL (c) Useop Gim 2022
# Please let reference
# If you intereseted in my work visit in
# https://sites.google.com/view/gesope/projects/a-i/neural-network-algorithm-explanation?authuser=0


# basic shuffle random
def REPLAY_SHUFFLE(replay_buffer, rl_data_dict):
    """
    It substitutes "index_list" of rl_data_dict

    Args:
        replay_buffer(np.ndarray): replay buffer array
        rl_data_dict(dict): dictionary for reinforcement learning data
    """
    index_list = np.arange(len(replay_buffer))
    np.random.shuffle(index_list)
    rl_data_dict["index_list"] = index_list


# prioritization
def REPLAY_PRIORITIZATION(replay_buffer, rl_data_dict) -> np.ndarray:
    """
    This is reordering the replay buffer index through
    the values with the largest difference are listed with the highest probability with annealing bias

    Args:
        replay_buffer(np.ndarray): replay buffer array
        rl_data_dict(dict): dictionary for reinforcement learning data

    Return:
        re-ordered index array
    """
    assert "rl_model" in rl_data_dict
    robust_priority = np.empty(0)
    for exp in replay_buffer:
        err = rl_data_dict["rl_model"](exp, rl_data_dict)["E"]
        robust_priority = np.append(robust_priority, err)
    ordered_index = np.argsort(robust_priority)[::-1]  # ordered index by error distance
    prob_replay = 1 / np.arange(1, len(replay_buffer) + 1, 1)  # robust probability
    annealing_bias = (1 / len(replay_buffer) * prob_replay) ** 0.5  # robust probability and annealing the bias
    prob_random = np.random.uniform(0, 1, len(replay_buffer))  # random probability list
    prob_picked = np.argsort(np.where(prob_random < annealing_bias, 0, 1))  # sorting picked probabilities
    return ordered_index[prob_picked]  # ordered index by sorting picked probabilities


# hindsight_replay
def REPLAY_HINDSIGHT(replay_buffer, rl_data_dict):
    """
    This function set the new goal by the last replay buffer
    After that re-calculating following trajectory again base on reward policy
    Then add those all trajectory into the replay buffer
    (if the size is over the oldest memory is deleted)

    Args:
        replay_buffer(np.ndarray): replay buffer array (object reference)
        rl_data_dict(dict): dictionary for reinforcement learning data
    """
    __last = replay_buffer[-1]  # get last
    __goal = __last[3]  # set new goal
    for trajectory in replay_buffer:
        __s, __a, __r, __ss, __t = trajectory  # the replay buffer
        __r, __t = rl_data_dict["reward_fn"](__s, __goal)  # change reward
        replay_buffer = RL_ADD_EXP(np.array([__s, __a, __r, __ss, __t], dtype=object),
                                   replay_buffer, rl_data_dict["replay_buffer_max_sz"])


# slice the replay buffer by replay size [return call by object reference]
def RL_TRG_UPDATE(t_update_step, rl_data_dict):
    """
    Update the target networks in rl_data_dict by rl_data_dict["t_update_interval"]

    Args:
        t_update_step(int): iteration for target Q value update
        rl_data_dict(dict): dictionary for reinforcement learning data
    """
    if t_update_step % rl_data_dict["t_update_interval"] == 0:
        if "tqn" in rl_data_dict:
            if rl_data_dict["t_update_rate"] != 1:
                rl_data_dict["tqn"].set_w_layer(rl_data_dict["qn"].get_layer()[0] * rl_data_dict["t_update_rate"] \
                                                + rl_data_dict["tqn"].get_layer()[0] * (
                                                        1.0 - rl_data_dict["t_update_rate"]))
                rl_data_dict["tqn"].set_b_layer(rl_data_dict["qn"].get_layer()[1] * rl_data_dict["t_update_rate"] \
                                                + rl_data_dict["tqn"].get_layer()[1] * (
                                                        1.0 - rl_data_dict["t_update_rate"]))
            else:
                rl_data_dict["tqn"].set_w_layer(rl_data_dict["qn"].get_layer()[0])
                rl_data_dict["tqn"].set_b_layer(rl_data_dict["qn"].get_layer()[1])
        if "tqn_2" in rl_data_dict:
            if rl_data_dict["t_update_rate"] != 1:
                rl_data_dict["tqn_2"].set_w_layer(rl_data_dict["qn_2"].get_layer()[0] * rl_data_dict["t_update_rate"] \
                                                  + rl_data_dict["tqn_2"].get_layer()[0] * (
                                                          1.0 - rl_data_dict["t_update_rate"]))
                rl_data_dict["tqn_2"].set_b_layer(rl_data_dict["qn_2"].get_layer()[1] * rl_data_dict["t_update_rate"] \
                                                  + rl_data_dict["tqn_2"].get_layer()[1] * (
                                                          1.0 - rl_data_dict["t_update_rate"]))
            else:
                rl_data_dict["tqn_2"].set_w_layer(rl_data_dict["qn_2"].get_layer()[0])
                rl_data_dict["tqn_2"].set_b_layer(rl_data_dict["qn_2"].get_layer()[1])


def ARG_MAXQ(status, neural, rl_data_dict):
    argmax_list = np.empty(rl_data_dict["act_sz"])
    for i, act in enumerate(rl_data_dict["act_list"]):
        argmax_list[i] = neural.run(np.append(status, act))
    return np.argmax(argmax_list)


def ARG_MINQ(status, neural, rl_data_dict):
    argmin_list = np.empty(rl_data_dict["act_sz"])
    for i, act in enumerate(rl_data_dict["act_list"]):
        argmin_list[i] = neural.run(np.append(status, act))
    return np.argmin(argmin_list)


def MAX_Q(status, neural, rl_data_dict):
    max_list = np.empty(rl_data_dict["act_sz"])
    for i, act in enumerate(rl_data_dict["act_list"]):
        max_list[i] = neural.run(np.append(status, act))
    return np.max(max_list)


def MEAN_Q(status, neural, rl_data_dict):
    mean_list = np.empty(rl_data_dict["act_sz"])
    for i, act in enumerate(rl_data_dict["act_list"]):
        mean_list[i] = neural.run(np.append(status, act))
    return np.max(mean_list)


def VA_Q(status, neural, rl_data_dict) -> float:
    max_list = np.empty(rl_data_dict["act_sz"])
    for i, act in enumerate(rl_data_dict["act_list"]):
        neural.run(np.append(status, act))
        max_list[i] = neural.output[1] + neural.output[0]
    return np.max(max_list)


def DQN(exp, rl_data_dict) -> dict:
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["merge"]:
        if not t:
            yt = r + gamma * MAX_Q(_s, rl_data_dict["tqn"], rl_data_dict)
        else:
            yt = r
        q = rl_data_dict["qn"].run(np.append(s, a))
    else:
        if not t:
            yt = r + gamma * max(rl_data_dict["tqn"].run(_s))
        else:
            yt = r
        q = rl_data_dict["qn"].run(s)[a]
    return {"Y": yt, "Q": q, "E": 0.5 * (yt - q)}


def DDQN(exp, rl_data_dict) -> dict:
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["merge"]:
        assert rl_data_dict["qn"].get_shape()[-1] == 1
        if not t:
            yt = r + gamma * MAX_Q(_s, rl_data_dict["qn"], rl_data_dict)
        else:
            yt = r
        q = rl_data_dict["qn"].run(np.append(s, a))
    else:
        if not t:
            yt = r + gamma * rl_data_dict["tqn"].run(_s)[np.argmax(rl_data_dict["qn"].run(_s))]
        else:
            yt = r
        q = rl_data_dict["qn"].run(s)[a]
    return {"Y": yt, "Q": q, "E": 0.5 * (yt - q)}


def D2QN(exp, rl_data_dict) -> dict:
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["merge"]:
        assert rl_data_dict["qn"].get_shape()[-1] == 1
        if not t:
            yt = r + gamma * (VA_Q(_s, rl_data_dict["tqn"], rl_data_dict) - MEAN_Q(_s, rl_data_dict["tqn"], rl_data_dict))
        else:
            yt = r
        VA = np.sum(rl_data_dict["qn"].run(np.append(s, a)))
        q = VA - MEAN_Q(s, rl_data_dict["qn"], rl_data_dict)
    else:
        if not t:
            rl_data_dict["tqn"].run(_s)
            yt = r + gamma * (rl_data_dict["tqn"].output[-1]  # V
                 + np.max(rl_data_dict["tqn"].output[:-1])  # A
                 - np.mean(rl_data_dict["tqn"].output[:-1]))  # mean A
        else:
            yt = r
        rl_data_dict["qn"].run(s)
        q = (rl_data_dict["qn"].output[-1]  # V
             + rl_data_dict["qn"].output[a]  # A
             - np.mean(rl_data_dict["qn"].output[:-1]))  # mean A
    return {"Y": yt, "Q": q, "E": 0.5 * (yt - q)}


def D3QN(exp, rl_data_dict) -> dict:
    """
     "Y" - r + gamma * ((V:Q'(s)) + (A:Q'(s))[argmax(Q(s')] + μ(A:Q'(s)))\n
     "Q" - ((V:Q(s)) + (A:Q(s))[a] + μ(A:Q(s)))\n
     "E" - 1/2(Y-Q)\n

     Args:
         exp(np.ndarray): the single array [s,a,r,s',t]
         rl_data_dict(dict): dictionary for reinforcement learning data

     Return:
         dictionary "Y, "Q", "E"
     """
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["merge"]:
        assert rl_data_dict["qn"].get_shape()[-1] == 1
        if not t:
            yt = r + gamma * (VA_Q(_s, rl_data_dict["tqn"], rl_data_dict) - MEAN_Q(_s, rl_data_dict["tqn"], rl_data_dict))
        else:
            yt = r
        VA = np.sum(rl_data_dict["qn"].run(np.append(s, a)))
        q = VA - MEAN_Q(s, rl_data_dict["qn"], rl_data_dict)
    else:
        if not t:
            rl_data_dict["tqn"].run(_s)
            yt = r + gamma * (rl_data_dict["tqn"].output[-1]  # V
                 + rl_data_dict["tqn"].output[:-1][np.argmax(rl_data_dict["qn"].run(_s))]  # A
                 - np.mean(rl_data_dict["tqn"].output[:-1]))  # mean A
        else:
            yt = r
        rl_data_dict["qn"].run(s)
        q = (rl_data_dict["qn"].output[-1]  # V
             + rl_data_dict["qn"].output[a]  # A
             - np.mean(rl_data_dict["qn"].output[:-1]))  # mean A
    return {"Y": yt, "Q": q, "E": 0.5 * (yt - q)}


def RL_ACTION(s, epsilon, rl_data_dict) -> int:
    """
    Args:
        s(np.ndarray) : current state
        act_sz(int) : action size
        epsilon(float) : between 0 and 1 portability
    Return:
        action(int): action from fn(s)
    """
    if rl_data_dict["merge"]:
        if 0 < epsilon and random.uniform(0, 1) < epsilon:
            return rl_data_dict["act_list"][random.randint(0, rl_data_dict["act_sz"] - 1)]
        return int(ARG_MAXQ(s, rl_data_dict["qn"], rl_data_dict))
    else:
        if 0 < epsilon and random.uniform(0, 1) < epsilon:
            return random.randint(0, rl_data_dict["act_sz"] - 1)
        return int(np.argmax(rl_data_dict["agent"].run(s)[:rl_data_dict["act_sz"]]))


def RL_ADD_EXP(exp, replay_buffer, rl_data_dict) -> np.ndarray:
    """
    Args:
        exp(np.ndarray(5, dtype = object)): ['state', 'action', 'reward', 'future state', 'termination']
        replay_buffer(np.ndarray) : replay_buffer
        replay_buffer_max_sz(int) : replay_buffer maximum size
    """
    if len(replay_buffer) + 1 > rl_data_dict["replay_buffer_max_sz"]:
        replay_buffer = np.delete(replay_buffer, 0, axis=0)
    s, a, r, _s, t = exp
    replay_buffer = np.vstack((replay_buffer,
                               np.array([s, a, r, _s, t], dtype=object)))
    return replay_buffer


def RL_REPLAY_SAC_LEARN(replay_buffer, rl_data_dict):
    assert rl_data_dict["replay_sz"] == 1
    for trial in range(rl_data_dict["replay_trial"]):
        # ---------SGD-----------SS
        for exp in replay_buffer:
            loss_data = rl_data_dict["rl_model"](exp=exp, rl_data_dict=rl_data_dict)
            Q = loss_data["Q"]
            Y = loss_data["Y"]
            P = loss_data["P"]
            PY = loss_data["PY"]
        rl_data_dict["qn"].learn_start(Q, Y)
        rl_data_dict["pn"].learn_start(P, PY)

def RL_MINI_BATCH_LEARN(replay_buffer, rl_data_dict):
    for trial in range(rl_data_dict["replay_trial"]):
        # ---------mini batch split-----------
        mini_batch = \
            replay_buffer[rl_data_dict["replay_sz"] * trial: rl_data_dict["replay_sz"] * (trial + 1)]
        # ---------mini batch replay----------
        Y = np.zeros(rl_data_dict["qn"].get_shape()[-1])
        Q = np.zeros(rl_data_dict["qn"].get_shape()[-1])
        # ---------accumulate gradient---------
        if len(mini_batch) >= rl_data_dict["replay_sz"]:
            for exp in mini_batch:
                loss_data = rl_data_dict["rl_model"](exp=exp, rl_data_dict=rl_data_dict)
                Y[exp[1]] += loss_data["Y"]
                Q[exp[1]] += loss_data["Q"]
            rl_data_dict["qn"].learn_start(Q, Y)
        else:
            break

def RL_REPLAY(replay_buffer, update_step, rl_data_dict):
    """
    Args:
        replay_buffer(np.ndarray): replay buffer
        update_step(int): update step
        rl_data_dict(dict): infomration dictionary
    """
    assert "replay_buffer_max_sz" in rl_data_dict
    assert "replay_sz" in rl_data_dict
    assert "replay_trial" in rl_data_dict
    assert "replay_opt" in rl_data_dict
    assert "w_update_interval" in rl_data_dict
    assert "rl_model" in rl_data_dict

    if len(replay_buffer) == rl_data_dict["replay_buffer_max_sz"] and update_step % rl_data_dict[
        "w_update_interval"] == 0:
        # ----------ordering replay----------
        replay_index_list = rl_data_dict["replay_opt"](replay_buffer, rl_data_dict)
        re_ordered_replay = replay_buffer[replay_index_list] # deepcopy array
        if rl_data_dict["rl_model"] == SAC:
            RL_REPLAY_SAC_LEARN(re_ordered_replay, rl_data_dict)
        else:
            RL_MINI_BATCH_LEARN(re_ordered_replay, rl_data_dict)


class openRL:
    """
    This is DQN class.
    After declare, following below process\n
    RL_DQN_SETTING : Make the neural network
    """

    __up_step = 0
    __reward_terminate_max = 12
    reward_time_stamp = np.array(0)

    qn = openNeural()  # value neural
    tqn = openNeural()  # target value neural

    __replay_buffer = np.ndarray  # structure as [state, action, reward, future, termination, TDerror]
    __replay_index_list = np.empty((0, 6))

    RL_DATA = {}

    def __init__(self, model, action_size, random_seed, act_list = None, replay_opt=REPLAY_PRIORITIZATION):
        """
        It initializes tje target and main neural network
        Args:
            model(function) : DQN, DDQN, D2QN, D3QN
            exp_opt(function) : Replay optimization, default is REPLAY_SHUFFLE, REPLAY_PRIORITIZATION
            action_size(int) : the nubme of action size
            random_seed(int) : the random seed
        """
        assert 0 < action_size
        np.random.seed(seed=random_seed)  # for testing
        random.seed(random_seed)

        self.RL_DATA["tqn"] = openNeural()
        self.RL_DATA["qn"] = openNeural()
        self.RL_DATA["rl_model"] = model
        self.RL_DATA["replay_opt"] = replay_opt
        self.RL_DATA["act_sz"] = action_size
        self.RL_DATA["merge"] = False
        self.RL_DATA["act_list"] = act_list

    def RL_RUN(self, initial_state, terminate_reward_condition=None, ep_decay_rate=0.9, show=True):
        """
        Args:
            initial_state(np.ndarray) : initial state
            terminate_reward_condition(float) : terminate_reward_condition if None the program do until max epoch
            ep_decay_rate(float) : epsilon decay rate
            show(bool): show each reward
        """
        __update = 0
        # ----------playing episode----------
        max_reward = 0
        total_reward = 0
        action_probability = 0.8
        reward_threshold = 1
        self.RL_DATA["qn"].opt_reset()
        self.RL_DATA["tqn"].opt_reset()
        self.RL_DATA["agent"] = self.RL_DATA["qn"]
        __replay_buffer = np.empty((0, 5))
        __up_step = 0
        for __ep in range(self.RL_DATA["max_epoch"]):
            # ----------playing----------
            s = initial_state.copy()  # initialize state S
            max_reward = max(total_reward, max_reward)
            total_reward = 0
            for __it in range(self.RL_DATA["max_iter"]):  # increase the interation
                __up_step += 1
                # ----------Get exp from action----------
                r, t = self.RL_DATA["reward_fn"](s)  # get [r, t] (reward and terminated state)
                a = RL_ACTION(s=s, epsilon=action_probability, rl_data_dict=self.RL_DATA)  # get a (action)
                ss = self.RL_DATA["enviro_fn"](s, a)  # get ss (new state)
                exp = np.array([s, a, r, ss, t], dtype=object)
                # ----------Add experience buffer----------
                __replay_buffer = RL_ADD_EXP(exp, __replay_buffer, self.RL_DATA)
                # ----------replay learning------
                RL_REPLAY(replay_buffer=__replay_buffer, update_step=__up_step, rl_data_dict=self.RL_DATA)
                # ------target uupdate--------------------
                RL_TRG_UPDATE(__up_step, rl_data_dict=self.RL_DATA)
                # ---------e-greedy policy -----------------
                total_reward += r
                # ---------Info -----------------
                if total_reward >= reward_threshold:
                    action_probability *= ep_decay_rate
                    reward_threshold += 0.5
                    action_probability = max(action_probability, 0.01)
                    print(__ep, __it, total_reward, action_probability)
                # ---------Termination check -----------------
                if t: break
                s = ss.copy()  # update new state
            self.reward_time_stamp = np.append(self.reward_time_stamp, total_reward)
            if terminate_reward_condition is not None and total_reward >= terminate_reward_condition:
                break

    def RL_LEARN_SETTING(self,
                         enviro_fn=None, reward_fn=None, max_iter=60, max_epoch=1000,
                         buffer_maximum_sz=32, buffer_replay_sz=1, buffer_replay_trial=1,
                         dropout_rate=0.0,
                         learning_rate=0.005, learn_optima='NADAM', loss_fun=HUBER,
                         w_update_interval=5, t_update_interval=10, gamma=0.99, t_update_rate=0.01
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
        assert 0.0 < t_update_rate <= 1.0
        assert 0.0 < learning_rate
        assert 0.0 <= dropout_rate <= 1
        assert str(type(loss_fun)) == "<class 'function'>"

        self.__replay_buffer = np.empty((0, 6))

        self.RL_DATA["tqn"] = openNeural()
        self.RL_DATA["qn"] = openNeural()

        self.RL_DATA["enviro_fn"] = enviro_fn
        self.RL_DATA["reward_fn"] = reward_fn
        self.RL_DATA["gamma"] = gamma
        self.RL_DATA["max_iter"] = max_iter
        self.RL_DATA["max_epoch"] = max_epoch
        self.RL_DATA["replay_buffer_max_sz"] = buffer_maximum_sz
        self.RL_DATA["replay_sz"] = buffer_replay_sz
        self.RL_DATA["replay_trial"] = buffer_replay_trial
        self.RL_DATA["w_update_interval"] = w_update_interval
        self.RL_DATA["t_update_rate"] = t_update_rate
        self.RL_DATA["t_update_interval"] = t_update_interval
        self.RL_DATA["qn"].opt_reset()
        self.RL_DATA["qn"].learning_set(
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
        # checking exception
        if q_layer[-1] == 1:
            self.RL_DATA["merge"] = True
            if len(self.RL_DATA["act_list"]) != self.RL_DATA["act_sz"]:
                raise Exception("act_size output and act_list size doesn't match")
            print("Merge input neural network is set")
        elif self.RL_DATA["rl_model"] == D2QN or self.RL_DATA["rl_model"] == D3QN:
            if not self.RL_DATA["merge"] and q_layer[-1] != self.RL_DATA["act_sz"] + 1:
                raise Exception("D2QN D3QN output size must act size + 1")
        elif self.RL_DATA["rl_model"] == DQN or self.RL_DATA["rl_model"] == DDQN:
            if not self.RL_DATA["merge"] and q_layer[-1] != self.RL_DATA["act_sz"]:
                raise Exception("DQN DDQN output size must as same as act size")


        self.RL_DATA["merging"] = 1
        # layer setting
        for i, L in enumerate(q_layer):
            self.RL_DATA["qn"].add_layer(L, q_activation_fn[i], normal=q_normalization[i])
        self.RL_DATA["qn"].generate_weight()
        self.RL_DATA["qn"].he_initialization()
        print('Q NEURAL : ', self.RL_DATA["qn"].get_shape())
        self.RL_DATA["tqn"] << self.RL_DATA["qn"]
        print('Q TARGET NEURAL : ', self.RL_DATA["tqn"].get_shape())

    def RL_POLICY_SETTING(self,
                          p_layer=np.array([4, 8, 12, 3]),
                          p_activation_fn=
                          np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                          p_normalization=
                          np.array([linear_x, znormal, znormal, linear_x], dtype=object)
                          ):
        if self.RL_DATA["rl_model"] != SAC:
            raise Exception("rl_model type must be Policy type")
        for i, L in enumerate(p_layer):
            self.RL_DATA["pn"].add_layer(L, p_activation_fn[i], normal=p_normalization[i])
        self.RL_DATA["pn"].generate_weight()
        self.RL_DATA["pn"].he_initialization()
        print('P NEURAL : ', self.RL_DATA["pn"].get_shape())
        self.RL_DATA["qn_2"] << self.RL_DATA["qn"]
        self.RL_DATA["qn_2"].he_initialization()
        self.RL_DATA["tqn_2"] << self.RL_DATA["qn_2"]
        print('Q_2 NEURAL : ', self.RL_DATA["qn_2"].get_shape())
        print('Q_2 TARGET NEURAL : ', self.RL_DATA["qn_2"].get_shape())

    def RL_SAVE(self, file_name) -> None:
        self.RL_DATA["qn"].numpy_save(file_name + "qn")
        self.RL_DATA["tqn"].numpy_save(file_name + "tqn")
        np.savetxt(file_name + "_act_list", self.RL_DATA["act_list"], fmt="%d")
        np.savetxt(file_name + "_act_sz", np.array([self.RL_DATA["act_sz"]]), fmt="%d")
        print("SAVE FINISH")

    def RL_LOAD(self, file_name) -> None:
        self.RL_DATA["tqn"].numpy_load(file_name + "tqn")
        self.RL_DATA["qn"].numpy_load(file_name + "qn")
        self.RL_DATA["act_list"] = np.loadtxt(file_name + "_act_list")
        self.RL_DATA["act_sz"] = np.loadtxt(file_name + "_act_sz", dtype = int)
        self.RL_DATA["agent"] = self.RL_DATA["qn"]
        if self.RL_DATA["qn"].get_shape()[-1] == 1:
            self.RL_DATA["merge"] = True
        print("LOAD FINISH")
