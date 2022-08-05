import numpy as np

from openRL_method import *
import os


# GNU AFFERO GPL (c) Useop Gim 2022
# Please let reference
# If you intereseted in my work visit in
# https://sites.google.com/view/gesope/projects/a-i/neural-network-algorithm-explanation?authuser=0

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

    def __init__(self):
        pass

    def RL_SETTING(self,
                   rl_model, enviro_fn, reward_fn, act_list,
                   max_epoch, max_iter,
                   replay_buffer_max_sz, replay_sz, replay_trial, replay_opt,
                   gamma, alpha, agent_update_interval, t_update_interval, t_update_rate):
        # -------RL setting-----------
        self.RL_DATA["rl_model"] = rl_model
        self.RL_DATA["enviro_fn"] = enviro_fn
        self.RL_DATA["reward_fn"] = reward_fn
        self.RL_DATA["act_list"] = act_list
        # ---------replay-------------
        self.RL_DATA["max_epoch"] = max_epoch
        self.RL_DATA["max_iter"] = max_iter
        self.RL_DATA["replay_buffer_max_sz"] = replay_buffer_max_sz
        self.RL_DATA["replay_sz"] = replay_sz
        self.RL_DATA["replay_trial"] = replay_trial
        if replay_opt is None:
            self.RL_DATA["replay_opt"] = REPLAY_SHUFFLE
        else:
            self.RL_DATA["replay_opt"] = replay_opt
        # ---------learning-----------
        self.RL_DATA["gamma"] = gamma
        self.RL_DATA["alpha"] = alpha
        self.RL_DATA["agent_update_interval"] = agent_update_interval
        self.RL_DATA["t_update_interval"] = t_update_interval
        self.RL_DATA["t_update_rate"] = t_update_rate
        self.RL_DATA["qn"] = np.empty(0)
        self.RL_DATA["tqn"] = np.empty(0)

    def CREATE_Q(self,
                 learning_rate, dropout_rate, loss_fun, learn_optima,
                 q_layer=np.array([4, 8, 12, 3]),
                 q_activation_fn=
                 np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                 q_normalization=
                 np.array([linear_x, znormal, znormal, linear_x], dtype=object),
                 ):
        # layer setting
        q_network = openNeural()
        for i, L in enumerate(q_layer):
            q_network.add_layer(L, q_activation_fn[i], normal=q_normalization[i])
        q_network.generate_weight()
        q_network.opt_reset()
        q_network.learning_set(learning_rate=learning_rate,
                               dropout_rate=dropout_rate,
                               loss_fun=loss_fun,
                               learn_optima=learn_optima)
        # target neural network
        tq_network = deepcopy(q_network)
        self.RL_DATA["qn"] = np.append(self.RL_DATA["qn"], q_network)
        self.RL_DATA["tqn"] = np.append(self.RL_DATA["tqn"], tq_network)

    def CREATE_P(self,
                 learning_rate, dropout_rate, loss_fun, learn_optima,
                 p_layer=np.array([4, 8, 12, 3]),
                 p_activation_fn=
                 np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                 p_normalization=
                 np.array([linear_x, znormal, znormal, linear_x], dtype=object),
                 ):
        # layer setting
        p_network = openNeural()
        for i, L in enumerate(p_layer):
            p_network.add_layer(L, p_activation_fn[i], normal=p_normalization[i])
        p_network.generate_weight()
        p_network.opt_reset()
        p_network.learning_set(learning_rate=learning_rate,
                               dropout_rate=dropout_rate,
                               loss_fun=loss_fun,
                               learn_optima=learn_optima)
        self.RL_DATA["pn"] = p_network

    def RL_NEURAL_SETTING(self):
        # --------Neural setting------
        if self.RL_DATA["rl_model"] is SAC:
            assert len(self.RL_DATA["qn"]) == 2
            self.RL_DATA["agent"] = self.RL_DATA["pn"]  # call by object reference
            self.RL_DATA["SA_merge"] = True
            self.RL_DATA["rl_learn"] = RL_ON_POLICY_LEARN
        else:
            self.RL_DATA["agent"] = self.RL_DATA["qn"][0]  # call by object reference
            self.RL_DATA["rl_learn"] = RL_OFF_POLICY_LEARN
        if self.RL_DATA["rl_model"] is D2QN or D3QN:
            if self.RL_DATA["qn"][0].get_shape()[-1] == 2:
                self.RL_DATA["SA_merge"] = True
            else:
                self.RL_DATA["SA_merge"] = False
        else:
            if self.RL_DATA["agent"].get_shape()[-1] == 1:
                self.RL_DATA["SA_merge"] = True
            else:
                self.RL_DATA["SA_merge"] = False

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
        replay_buffer = np.empty((0, 5))
        update_step = 0
        for __ep in range(self.RL_DATA["max_epoch"]):
            # ----------playing----------
            s = initial_state.copy()  # initialize state S
            max_reward = max(total_reward, max_reward)
            total_reward = 0
            for __it in range(self.RL_DATA["max_iter"]):  # increase the interation
                update_step += 1
                # ----------Get exp from action----------
                r, t = self.RL_DATA["reward_fn"](s)  # get [r, t] (reward and terminated state)
                a = RL_ACTION(s=s,
                              epsilon=action_probability,
                              agent=self.RL_DATA["agent"],
                              act_list=self.RL_DATA["act_list"],
                              SA_merge=self.RL_DATA["SA_merge"])  # get a (action)
                ss = self.RL_DATA["enviro_fn"](s, a)  # get ss (new state)
                exp = np.array([s, a, r, ss, t], dtype=object)
                # ----------Add experience buffer----------
                replay_buffer = RL_ADD_EXP(exp=exp,
                                           replay_buffer=replay_buffer,
                                           replay_buffer_max_sz=self.RL_DATA["replay_buffer_max_sz"])
                # ----------replay learning------
                RL_REPLAY(replay_buffer=replay_buffer, update_step=update_step, rl_data_dict=self.RL_DATA)
                # ------target uupdate--------------------
                RL_TRG_UPDATE(update_step, rl_data_dict=self.RL_DATA)
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

    def RL_SAVE(self, file_name) -> None:
        os.makedirs("save", exist_ok=True)
        setting_file = np.array([
            self.RL_DATA["rl_model"],
            self.RL_DATA["enviro_fn"],
            self.RL_DATA["reward_fn"],
            self.RL_DATA["act_list"],
            self.RL_DATA["max_epoch"],
            self.RL_DATA["max_iter"],
            self.RL_DATA["replay_buffer_max_sz"],
            self.RL_DATA["replay_sz"],
            self.RL_DATA["replay_trial"],
            self.RL_DATA["replay_opt"],
            self.RL_DATA["gamma"],
            self.RL_DATA["alpha"],
            self.RL_DATA["agent_update_interval"],
            self.RL_DATA["t_update_interval"],
            self.RL_DATA["t_update_rate"],
            len(self.RL_DATA["qn"]), len(self.RL_DATA["tqn"])], dtype=object)
        np.save("save/" + file_name + "_info", setting_file, allow_pickle=True)
        # ----------------------------------------------------
        for i, qn in enumerate(self.RL_DATA["qn"]):
            qn.numpy_save("save/" + file_name + "_qn_" + str(i))
        for i, tqn in enumerate(self.RL_DATA["tqn"]):
            tqn.numpy_save("save/" + file_name + "_tqn_" + str(i))
        try:
            self.RL_DATA["pn"].numpy_save("save/" + file_name + "_pn")
        except Exception as e:
            pass
        print("SAVE FINISH")

    def RL_LOAD(self, file_name) -> None:

        [self.RL_DATA["rl_model"],
         self.RL_DATA["enviro_fn"],
         self.RL_DATA["reward_fn"],
         self.RL_DATA["act_list"],
         self.RL_DATA["max_epoch"],
         self.RL_DATA["max_iter"],
         self.RL_DATA["replay_buffer_max_sz"],
         self.RL_DATA["replay_sz"],
         self.RL_DATA["replay_trial"],
         self.RL_DATA["replay_opt"],
         self.RL_DATA["gamma"],
         self.RL_DATA["alpha"],
         self.RL_DATA["agent_update_interval"],
         self.RL_DATA["t_update_interval"],
         self.RL_DATA["t_update_rate"],
         qn_sz, tqn_sz] = np.load("save/" + file_name + "_info.npy", allow_pickle=True)
        # ----------------------------------------------------
        for i in range(qn_sz):
            q_network = openNeural()
            q_network.numpy_load("save/" + file_name + "_qn_" + str(i))
            self.RL_DATA["qn"] = np.append(self.RL_DATA["qn"], q_network)
        for i in range(tqn_sz):
            q_network = openNeural()
            q_network.numpy_load("save/" + file_name + "_tqn_" + str(i))
            self.RL_DATA["tqn"] = np.append(self.RL_DATA["tqn"], q_network)
        try:
            self.RL_DATA["pn"].numpy_load("save/" + file_name + "_pn")
        except Exception as e:
            pass
        self.RL_NEURAL_SETTING()

        print("LOAD FINISH")
