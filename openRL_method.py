import random

import numpy as np

from openNeural import *


# base on the all actions, finding the argmax Q
def ARG_MAXQ(status, neural, act_list) -> int:
    argmax_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        argmax_list[i] = neural.run(np.append(status, act))[0]
    return int(np.argmax(argmax_list))


# base on the all actions, finding the argmin Q
def ARG_MINQ(status, neural, act_list) -> int:
    argmin_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        argmin_list[i] = neural.run(np.append(status, act))[0]
    return int(np.argmin(argmin_list))


# base on the all actions, finding the max Q
def MAX_Q(status, neural, act_list) -> float:
    max_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        max_list[i] = neural.run(np.append(status, act))[0]
    return np.max(max_list)


# base on the all actions, finding the mean Q
def MEAN_Q(status, neural, act_list) -> float:
    mean_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        mean_list[i] = neural.run(np.append(status, act))[0]
    return np.max(mean_list)


# base on the all actions, finding the Value and Advantage
def ARGMAX_VA_Q(status, neural, act_list) -> int:
    max_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        neural.run(np.append(status, act))
        max_list[i] = neural.output[1] + neural.output[0]
    return int(np.argmax(max_list))


# base on the all actions, finding the Value and Advantage
def MAX_VA_Q(status, neural, act_list) -> float:
    max_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        neural.run(np.append(status, act))
        max_list[i] = neural.output[1] + neural.output[0]
    return np.max(max_list)


# base on the all actions, finding all Q
def ALL_Q(status, neural, act_list) -> np.ndarray:
    action_candidates = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        neural.run(np.append(status, act))
        action_candidates[i] = neural.output[1]
    return action_candidates


# ------------------MODEL--------------------------

# call by object reference
def E_G_DECAY_BY_REWARD(rl_data_dict):
    assert "total_reward" in rl_data_dict
    assert "epsilon" in rl_data_dict
    assert "epsilon_decay_rate" in rl_data_dict
    assert "epsilon_decay_threshold" in rl_data_dict
    assert "epsilon_decay_threshold_rate" in rl_data_dict
    assert "epsilon_min" in rl_data_dict
    if rl_data_dict["total_reward"] >= rl_data_dict["epsilon_decay_threshold"]:
        rl_data_dict["epsilon"] *= rl_data_dict["epsilon_decay_rate"]
        rl_data_dict["epsilon_decay_threshold"] += rl_data_dict["epsilon_decay_threshold_rate"]
        rl_data_dict["epsilon"] = max(rl_data_dict["epsilon"], rl_data_dict["epsilon_min"])


def E_G_DECAY_BY_EPISODE(rl_data_dict):
    assert "max_epoch" in rl_data_dict
    assert "epsilon" in rl_data_dict
    rl_data_dict["epsilon"] = 1 - rl_data_dict["epoch"]/rl_data_dict["max_epoch"]


def DQN(exp, rl_data_dict) -> dict:
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        if not t:
            yt = r + gamma * MAX_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"])
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(np.append(s, a))
        a = 0
    else:
        if not t:
            yt = r + gamma * max(rl_data_dict["tqn"][0].run(_s))
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(s)[a]
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y,  "E": 0.5 * (yt - q)}


def DDQN(exp, rl_data_dict) -> dict:
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[-1] == 1
        if not t:
            yt = r + gamma * MAX_Q(_s, rl_data_dict["qn"][0], rl_data_dict["act_list"])
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(np.append(s, a))
        a = 0
    else:
        if not t:
            yt = r + gamma * rl_data_dict["tqn"][0].run(_s)[np.argmax(rl_data_dict["qn"][0].run(_s))]
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(s)[a]
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y, "E": 0.5 * (yt - q)}


def D2QN(exp, rl_data_dict) -> dict:
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[-1] == 2
        if not t:
            yt = r + gamma * (
                    MAX_VA_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"])
                    - MEAN_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"]))
        else:
            yt = r
        VA = np.sum(rl_data_dict["qn"][0].run(np.append(s, a)))
        q = VA - MEAN_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"])
        a = 0
    else:
        if not t:
            rl_data_dict["tqn"][0].run(_s)
            yt = r + gamma * (rl_data_dict["tqn"][0].output[-1]  # V
                              + np.max(rl_data_dict["tqn"][0].output[:-1])  # A
                              - np.mean(rl_data_dict["tqn"][0].output[:-1]))  # mean A
        else:
            yt = r
        rl_data_dict["qn"][0].run(s)
        q = (rl_data_dict["qn"][0].output[-1]  # V
             + rl_data_dict["qn"][0].output[a]  # A
             - np.mean(rl_data_dict["qn"][0].output[:-1]))  # mean A
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y, "E": 0.5 * (yt - q)}


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
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[-1] == 2
        if not t:
            yt = r + gamma * (
                    MAX_VA_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"])
                    - MEAN_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"]))
        else:
            yt = r
        VA = np.sum(rl_data_dict["qn"][0].run(np.append(s, a)))
        q = VA - MEAN_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"])
        a = 0
    else:
        if not t:
            rl_data_dict["tqn"][0].run(_s)
            yt = r + gamma * (rl_data_dict["tqn"][0].output[-1]  # V
                              + rl_data_dict["tqn"][0].output[:-1][np.argmax(rl_data_dict["qn"][0].run(_s)[:-1])]  # A
                              - np.mean(rl_data_dict["tqn"][0].output[:-1]))  # mean A
        else:
            yt = r
        rl_data_dict["qn"][0].run(s)
        q = (rl_data_dict["qn"][0].output[-1]  # V
             + rl_data_dict["qn"][0].output[a]  # A
             - np.mean(rl_data_dict["qn"][0].output[:-1]))  # mean A
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y, "E": 0.5 * (yt - q)}


def SAC(exp, rl_data_dict) -> dict:
    """
        Args:
            QN(list): [qn, qn_2, pn]
            TQN(list): [tqn, tqn_2]
            alpha_gamma(list): [alpha, gamma]
        """
    assert "gamma" in rl_data_dict
    assert "alpha" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["tqn"]) == len(rl_data_dict["qn"])
    assert rl_data_dict["SA_merge"] is True
    s, a, r, ss, t = exp
    alpha, gamma = rl_data_dict["alpha"], rl_data_dict["gamma"]
    pn, qn_1, qn_2 = rl_data_dict["qn"][0], rl_data_dict["qn"][1], rl_data_dict["qn"][2]
    tqn_1, tqn_2 = rl_data_dict["tqn"][0], rl_data_dict["tqn"][1]
    # -------------------------- State ----------------------------------------------------
    # --------------------- Select Index --------------------------------------------------
    t_q_ss_1_a = MAX_Q(ss, tqn_1, rl_data_dict["act_list"])  # TQ(s',a')
    t_q_ss_2_a = MAX_Q(ss, tqn_1, rl_data_dict["act_list"])  # TQ(s',a')
    I = np.argmin([t_q_ss_1_a, t_q_ss_2_a])  # Index
    q_i = [qn_1, qn_2][I]  # Q
    tq_i = [tqn_1, tqn_2][I]  # target Q
    # ----------------------Update Q value ------------------------------------------------
    # ------ V_s_1 = Q'(ss,a) - a*log(pi(ss)) # Update for Critic Network
    # ------ Jq = (r + g * V_s_1) - Q(s,a)
    v_ss = MAX_Q(ss, tq_i, rl_data_dict["act_list"]) - alpha * logp1_x(max(pn.run(ss)))
    if not t:
        y_ss = (r + gamma * v_ss)
    else:
        y_ss = r
    q = MAX_Q(s, q_i, rl_data_dict["act_list"])
    # ---------------------- Update policy -------------------------------------------------
    original = rl_data_dict["act_list"].copy()
    # ------ Jp = Q( s, f(p(s)) ) - a*log( f(p(s)) )
    re_parameterization = softmax(pn.run(s))
    p_l = alpha * logp1_x(max(re_parameterization))  # a*log( f(p(s)) )
    rl_data_dict["act_list"] = re_parameterization
    q_s_rp = MAX_Q(s, q_i, rl_data_dict["act_list"])  # Q( s, f(p(s)) )
    rl_data_dict["act_list"] = original
    # ---------------------- Update alpha function ------------------------------------------
    rl_data_dict["alpha"] = 0.5
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[I+1] = q
    Y[I+1] = y_ss
    Q[0][a] = p_l
    Y[0][a] = q_s_rp
    return {"Q": Q, "Y": Y, "E": shannon_entropy(p_l)}


# -------------------OPTIMIZATION-----------------------------


# ordering
def REPLAY_DIRECT(replay_buffer, rl_data_dict):
    return np.arange(len(replay_buffer))


# basic shuffle random
def REPLAY_SHUFFLE(replay_buffer, rl_data_dict):
    """
    It substitutes "index_list" of rl_data_dict

    Args:
        replay_buffer(np.ndarray): replay buffer array
        rl_data_dict(dict): dictionary for reinforcement learning data
    """
    assert len(replay_buffer) > 0
    index_list = np.arange(len(replay_buffer))
    np.random.shuffle(index_list)
    return index_list


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
    assert len(replay_buffer) > 0
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
        replay_buffer = RL_ADD_EXP(exp=np.array([__s, __a, __r, __ss, __t], dtype=object),
                                   replay_buffer=replay_buffer,
                                   replay_buffer_max_sz=rl_data_dict["replay_buffer_max_sz"])


# slice the replay buffer by replay size [return call by object reference]
def RL_TRG_UPDATE(t_update_step, rl_data_dict):
    """
    Update the target networks in rl_data_dict by rl_data_dict["t_update_interval"]

    Args:
        t_update_step(int): iteration for target Q value update
        rl_data_dict(dict): dictionary for reinforcement learning data
    """
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "t_update_rate" in rl_data_dict
    assert "t_update_interval" in rl_data_dict
    assert len(rl_data_dict["tqn"]) == len(rl_data_dict["qn"])

    if t_update_step % rl_data_dict["t_update_interval"] == 0:
        for qn, tqn in zip(rl_data_dict["qn"], rl_data_dict["tqn"]):
            if rl_data_dict["t_update_rate"] != 1:
                tqn.set_w_layer(qn.get_layer()[0] * rl_data_dict["t_update_rate"] \
                                + tqn.get_layer()[0] * (1.0 - rl_data_dict["t_update_rate"]))
                tqn.set_b_layer(qn.get_layer()[1] * rl_data_dict["t_update_rate"] \
                                + tqn.get_layer()[1] * (1.0 - rl_data_dict["t_update_rate"]))
            else:
                tqn.set_w_layer(qn.get_layer()[0])
                tqn.set_b_layer(qn.get_layer()[1])


# ------------------PROCESS--------------------------

def RL_E_G_ACTION(s, rl_data_dict) -> int:
    agent = rl_data_dict["agent"]
    act_list = rl_data_dict["act_list"]
    if 0 < rl_data_dict["epsilon"] and random.uniform(0, 1) < rl_data_dict["epsilon"]:
        return np.random.choice(act_list)
    else:
        if rl_data_dict["SA_merge"]:
            if rl_data_dict["rl_model"] is D2QN or rl_data_dict["rl_model"] is D3QN:
                return ARGMAX_VA_Q(s, agent, act_list)
            return ARG_MAXQ(s, agent, act_list)
    return int(np.argmax(agent.run(s)))


def RL_DIRECT_ACTION(s, rl_data_dict) -> int:
    agent = rl_data_dict["agent"]
    act_list = rl_data_dict["act_list"]
    if rl_data_dict["SA_merge"]:
        if rl_data_dict["rl_model"] is D2QN or D3QN:
            return ARGMAX_VA_Q(s, agent, act_list)
        return ARG_MAXQ(s, agent, act_list)
    return int(np.argmax(agent.run(s)))


def RL_ADD_EXP(exp, replay_buffer, replay_buffer_max_sz) -> np.ndarray:
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


def RL_LEARN(replay_buffer, rl_data_dict):
    for trial in range(rl_data_dict["replay_trial"]):
        # ---------mini batch split-----------
        mini_batch = \
            replay_buffer[rl_data_dict["replay_sz"] * trial: rl_data_dict["replay_sz"] * (trial + 1)]
        # ---------mini batch replay----------
        Q = np.empty(len(rl_data_dict["qn"]), dtype = list)
        for i, qn in enumerate(rl_data_dict["qn"]):
            Q[i] = np.zeros(qn.get_shape()[-1])
        Y = Q.copy()
        # ---------accumulate gradient---------
        if len(mini_batch) >= rl_data_dict["replay_sz"]:
            for exp in mini_batch:
                loss_data = rl_data_dict["rl_model"](exp=exp, rl_data_dict=rl_data_dict)
                Y += loss_data["Y"]
                Q += loss_data["Q"]
            for i, qn in enumerate(rl_data_dict["qn"]):
                qn.learn_start(Q[i], Y[i])
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
    assert "agent_update_interval" in rl_data_dict

    if len(replay_buffer) == rl_data_dict["replay_buffer_max_sz"] \
            and update_step % rl_data_dict["agent_update_interval"] == 0:
        # ----------ordering replay----------
        if rl_data_dict["replay_opt"] is None:
            replay_index_list = np.arange(len(replay_buffer))
        else:
            replay_index_list = rl_data_dict["replay_opt"](replay_buffer, rl_data_dict)
        re_ordered_replay = replay_buffer[replay_index_list]  # deepcopy array
        RL_LEARN(re_ordered_replay, rl_data_dict)
