import random
import numpy as np

from openNeural import *


# base on the all actions, finding the argmax Q
def ARG_MAXQ(status, neural, act_list, index=0) -> int:
    """

    :param status:
    :type status:np.ndarray or list
    :param neural:
    :type neural:openNeural
    :param act_list:
    :type act_list:np.ndarray or list
    """
    argmax_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        argmax_list[i] = neural.run(np.append(status, act))[index]
    return int(np.argmax(argmax_list))


# base on the all actions, finding the argmin Q
def ARG_MINQ(status, neural, act_list, index=0) -> int:
    """

    :param status:
    :type status:np.ndarray or list
    :param neural:
    :type neural:openNeural
    :param act_list:
    :type act_list:np.ndarray or list
    """
    argmin_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        argmin_list[i] = neural.run(np.append(status, act))[index]
    return int(np.argmin(argmin_list))


# base on the all actions, finding the max Q
def MAX_Q(status, neural, act_list, index=0) -> float:
    """

    :param status:
    :type status:np.ndarray or list
    :param neural:
    :type neural:openNeural
    :param act_list:
    :type act_list:np.ndarray or list
    """
    max_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        max_list[i] = neural.run(np.append(status, act))[index]
    return np.max(max_list)


# base on the all actions, finding the mean Q
def MEAN_Q(status, neural, act_list, index=0) -> float:
    """

    :param status:
    :type status:np.ndarray or list
    :param neural:
    :type neural:openNeural
    :param act_list:
    :type act_list:np.ndarray or list
    """
    mean_list = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        mean_list[i] = neural.run(np.append(status, act))[index]
    return np.max(mean_list)


# base on the all actions, finding all Q
def ALL_Q(status, neural, act_list, index=0) -> np.ndarray or list:
    """

    :param status:
    :type status:np.ndarray or list
    :param neural:
    :type neural:openNeural
    :param act_list:
    :type act_list:np.ndarray or list
    """
    action_candidates = np.empty(len(act_list))
    for i, act in enumerate(act_list):
        neural.run(np.append(status, act))
        action_candidates[i] = neural.output[index]
    return action_candidates


# ------------ epsion greedy --------------------


def E_G_DECAY_BY_REWARD(rl_data_dict) -> None:
    """

    :param rl_data_dict:
    :type rl_data_dict: dict
    """
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


def E_G_DECAY_BY_EPISODE(rl_data_dict) -> None:
    """

    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "max_epoch" in rl_data_dict
    assert "epsilon" in rl_data_dict
    rl_data_dict["epsilon"] = 1 - rl_data_dict["epoch"] / rl_data_dict["max_epoch"]


# --------------- rl_model ------------------------------------------

def DQN(exp, rl_data_dict) -> dict:
    """

    :param exp:
    :type exp: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["qn"]) == 1, "Only need single pair of Q and Target Q"

    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s) + 1, "Q(s,a) input size must |s| + 1"
        assert rl_data_dict["qn"][0].get_shape()[-1] == 1, "Q(s) out put size must 1"
        if not t:
            yt = r + gamma * MAX_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"])
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(np.append(s, a))
        a = 0
    else:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s), "Q(s) input size must |s|"
        assert rl_data_dict["qn"][0].get_shape()[-1] == len(rl_data_dict["act_list"]), "Q(s) out put size must |a|"
        if not t:
            yt = r + gamma * max(rl_data_dict["tqn"][0].run(_s))
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(s)[a]
    assert type(float(q)) is float
    assert type(float(yt)) is float
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y, "E": 0.5 * (yt - q)}


def DDQN(exp, rl_data_dict) -> dict:
    """

    :param exp:
    :type exp: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["qn"]) == 1, "Only need single pair of Q and Target Q"

    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s) + 1, "Q(s,a) input size must |s| + 1"
        assert rl_data_dict["qn"][0].get_shape()[-1] == 1, "Q(s) out put size must 1"
        if not t:
            _a = rl_data_dict["act_list"][ARG_MAXQ(_s, rl_data_dict["qn"][0], rl_data_dict["act_list"])]
            yt = r + gamma * rl_data_dict["tqn"][0].run(np.append(_s, _a))
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(np.append(s, a))
        a = 0
    else:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s), "Q(s) input size must |s|"
        assert rl_data_dict["qn"][0].get_shape()[-1] == len(rl_data_dict["act_list"]), "Q(s) out put size must |a|"
        if not t:
            yt = r + gamma * rl_data_dict["tqn"][0].run(_s)[np.argmax(rl_data_dict["qn"][0].run(_s))]
        else:
            yt = r
        q = rl_data_dict["qn"][0].run(s)[a]
    assert type(float(q)) is float
    assert type(float(yt)) is float
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[0][a] = q
    Y[0][a] = yt
    return {"Q": Q, "Y": Y, "E": 0.5 * (yt - q)}


def D2QN(exp, rl_data_dict) -> dict:
    """

    :param exp:
    :type exp: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["qn"]) == 1, "Only need single pair of Q and Target Q"

    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s) + 1, "Q(s,a) input size must |s| + 1"
        assert rl_data_dict["qn"][0].get_shape()[-1] == 2, "Value-Advantage output size must 2 "
        if not t:
            yt = r + gamma * (
                    ALL_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=1)[a]  # V
                    - ALL_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=0)[a]  # A
                    - MEAN_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=0))  # mean A
        else:
            yt = r
        q = (ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=1)[a]  # V
             + ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0)[a]  # A
             - MEAN_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0))  # mean A
        a = 0
    else:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s), "Q(s) input size must |s|"
        assert rl_data_dict["qn"][0].get_shape()[-1] == len(
            rl_data_dict["act_list"]) + 1, "Q(s) out put size must |a|+1"
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
    assert type(float(q)) is float
    assert type(float(yt)) is float
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

    :param exp:
    :type exp: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "gamma" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["qn"]) == 1, "Only need single pair of Q and Target Q"

    gamma = rl_data_dict["gamma"]
    s, a, r, _s, t = exp
    if rl_data_dict["SA_merge"]:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s) + 1, "Q(s,a) input size must |s| + 1"
        assert rl_data_dict["qn"][0].get_shape()[-1] == 2, "Value-Advantage output size must 2 "
        if not t:
            list_q = (MAX_Q(_s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=1)  # V
                      + MAX_Q(_s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0))  # Q
            _a = rl_data_dict["act_list"][np.argmax(list_q)]
            yt = r + gamma * (
                    ALL_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=1)[_a]  # V
                    - ALL_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=0)[_a]  # A
                    - MEAN_Q(_s, rl_data_dict["tqn"][0], rl_data_dict["act_list"], index=0))  # mean A
        else:
            yt = r
        q = (ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=1)[a]  # V
             + ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0)[a]  # A
             - MEAN_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0))  # mean A
        a = 0
    else:
        assert rl_data_dict["qn"][0].get_shape()[0] == len(s), "Q(s) input size must |s|"
        assert rl_data_dict["qn"][0].get_shape()[-1] == len(
            rl_data_dict["act_list"]) + 1, "Q(s) out put size must |a|+1"
        if not t:
            rl_data_dict["tqn"][0].run(_s)
            _a = np.argmax(rl_data_dict["qn"][0].run(_s)[:-1])
            yt = r + gamma * (rl_data_dict["tqn"][0].output[-1]  # V
                              + rl_data_dict["tqn"][0].output[:-1][_a]  # A
                              - np.mean(rl_data_dict["tqn"][0].output[:-1]))  # mean A
        else:
            yt = r
        rl_data_dict["qn"][0].run(s)
        q = (rl_data_dict["qn"][0].output[-1]  # V
             + rl_data_dict["qn"][0].output[a]  # A
             - np.mean(rl_data_dict["qn"][0].output[:-1]))  # mean A
    assert type(float(q)) is float
    assert type(float(yt)) is float
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

    :param exp:
    :type exp: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict:dict
    """
    assert "gamma" in rl_data_dict
    assert "alpha" in rl_data_dict
    assert "tqn" in rl_data_dict
    assert "qn" in rl_data_dict
    assert "SA_merge" in rl_data_dict
    assert len(rl_data_dict["qn"]) == 3, "SAC requires 3 q value and 2 target value"
    assert rl_data_dict["qn"][1].get_shape()[-1] == 1, "SAC 1st Q(s,a) requires single output"
    assert rl_data_dict["qn"][2].get_shape()[-1] == 1, "SAC 2nd Q(s,a) requires single output"
    s, a, r, ss, t = exp
    alpha, gamma = rl_data_dict["alpha"], rl_data_dict["gamma"]
    pn, qn_1, qn_2 = rl_data_dict["qn"][0], rl_data_dict["qn"][1], rl_data_dict["qn"][2]
    tqn_1, tqn_2 = rl_data_dict["tqn"][1], rl_data_dict["tqn"][2]
    # -------------------------- State ----------------------------------------------------
    # --------------------- Select Index --------------------------------------------------
    t_q_ss_1_a = MAX_Q(ss, tqn_1, pn.run(ss))  # TQ(s',a')
    t_q_ss_2_a = MAX_Q(ss, tqn_2, pn.run(ss))  # TQ(s',a')
    I = int(np.argmin([t_q_ss_1_a, t_q_ss_2_a])) + 1  # Index
    # ----------------------Update Q value ------------------------------------------------
    # ------ V_s_1 = Q'(ss,pi(ss)) - a*log(pi(ss)) # Update for Critic Network a' = pi(ss)
    # ------ Jq = (r + g * V_s_1) - Q(s,a)
    if not t:
        V = ALL_Q(s, rl_data_dict["qn"][I], pn.run(ss), index=0) - alpha * logp1_x(pn.run(ss))
        Y_ss = (r + gamma * V)[a]
    else:
        Y_ss = r
    QY = rl_data_dict["qn"][I].run(np.append(s, a))
    assert type(float(Y_ss)) is float
    assert type(float(QY)) is float
    # ---------------------- Update policy -------------------------------------------------
    # ------ Jp = Q( s, p(f(s)) ) - a*log( p(f(s)) )
    re_parameterization = re_parameterization_gaussian(s)
    P = alpha * logp1_x(pn.run(s))[a]  # a*log( p(f(s)) )
    Y_P = rl_data_dict["qn"][I].run(np.append(s, re_parameterization))  # Q( s, f(s) )
    assert type(float(P)) is float
    assert type(float(Y_P)) is float
    # ---------------------- Update alpha function ------------------------------------------
    rl_data_dict["alpha"] = 0.5
    # ---------------------------------- RETURN LOSS -----------------------------------------
    Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
    for i, qn in enumerate(rl_data_dict["qn"]):
        Q[i] = np.zeros(qn.get_shape()[-1])
    Y = Q.copy()
    Q[I] = Y_ss
    Y[I] = QY
    Q[0][a] = P
    Y[0][a] = Y_P
    return {"Q": Q, "Y": Y, "E": shannon_entropy(P)}


# -------------------OPTIMIZATION-----------------------------


# ordering
def REPLAY_DIRECT(replay_buffer, rl_data_dict) -> int:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    return int(np.arange(len(replay_buffer)))


# basic shuffle random
def REPLAY_SHUFFLE(replay_buffer, rl_data_dict) -> np.ndarray:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert len(replay_buffer) > 0, "Buffer size must be greater than 0"
    index_list = np.arange(len(replay_buffer))
    np.random.shuffle(index_list)
    return index_list


# prioritization
def REPLAY_PRIORITIZATION(replay_buffer, rl_data_dict) -> np.ndarray:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert len(replay_buffer) > 0, "Buffer size must be greater than 0"
    assert "rl_model" in rl_data_dict, "Reinforcement Model is required"
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
def REPLAY_HINDSIGHT(replay_buffer, rl_data_dict) -> None:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    __last = replay_buffer[-1]  # get last
    __goal = __last[3]  # set new goal
    for trajectory in replay_buffer:
        __s, __a, __r, __ss, __t = trajectory  # the replay buffer
        __r, __t = rl_data_dict["reward_fn"](__s, __goal)  # change reward
        replay_buffer = RL_ADD_EXP(exp=np.ndarray([__s, __a, __r, __ss, __t]),
                                   replay_buffer=replay_buffer,
                                   replay_buffer_max_sz=rl_data_dict["replay_buffer_max_sz"])


# slice the replay buffer by replay size [return call by object reference]
def RL_TRG_UPDATE(t_update_step, rl_data_dict) -> None:
    """

    :param t_update_step:
    :type t_update_step:int
    :param rl_data_dict:
    :type rl_data_dict: dict
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

def RL_ACTION(s, rl_data_dict) -> int:
    """

    :param s:
    :type s: np.ndarray or list
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    assert "agent" in rl_data_dict, "agent is required"
    assert "act_list" in rl_data_dict, "act_list is required"
    agent = rl_data_dict["agent"]
    if rl_data_dict["rl_model"] == SAC:
        return int(np.argmax(agent.run(s)))
    act_list = rl_data_dict["act_list"]
    if 0 < rl_data_dict["epsilon"] and random.uniform(0, 1) < rl_data_dict["epsilon"]:
        return np.random.choice(act_list)
    else:
        if rl_data_dict["SA_merge"]:
            assert rl_data_dict["agent"].get_shape()[0] == len(s) + 1, "Q(s,a) input size must |s| + 1"
            if rl_data_dict["rl_model"] is D2QN or D3QN:
                assert rl_data_dict["agent"].get_shape()[-1] == 2, "Value-Advantage output size must 2 "
                q = (ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=1)  # V
                     + ALL_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0)  # A
                     - MEAN_Q(s, rl_data_dict["qn"][0], rl_data_dict["act_list"], index=0))  # mean A
                return int(np.argmax(q))
            return ARG_MAXQ(s, agent, act_list, index=0)
    assert rl_data_dict["agent"].get_shape()[0] == len(s), "Q(s) input size must |s|"
    return int(np.argmax(agent.run(s)))


def RL_ADD_EXP(exp, replay_buffer, replay_buffer_max_sz) -> np.ndarray:
    """

    :param exp:
    :type exp: np.ndarray or list
    :param replay_buffer:
    :type replay_buffer: np.ndarray or List[list]
    :param replay_buffer_max_sz:
    :type replay_buffer_max_sz: int
    """
    if len(replay_buffer) + 1 > replay_buffer_max_sz:
        replay_buffer = np.delete(replay_buffer, 0, axis=0)
    s, a, r, _s, t = exp
    replay_buffer = np.vstack((replay_buffer,
                               np.array([s, a, r, _s, t], dtype=object)))
    return replay_buffer


def RL_LEARN(replay_buffer, rl_data_dict) -> None:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray or List[list]
    :param rl_data_dict:
    :type rl_data_dict: dict
    """
    for trial in range(rl_data_dict["replay_trial"]):
        # ---------mini batch split-----------
        mini_batch = \
            replay_buffer[rl_data_dict["replay_sz"] * trial: rl_data_dict["replay_sz"] * (trial + 1)]
        # ---------mini batch replay----------
        Q = np.empty(len(rl_data_dict["qn"]), dtype=list)
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


def RL_REPLAY(replay_buffer, update_step, rl_data_dict) -> None:
    """

    :param replay_buffer:
    :type replay_buffer: np.ndarray or list
    :param update_step:
    :type update_step: int
    :param rl_data_dict:
    :type rl_data_dict: dict
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
