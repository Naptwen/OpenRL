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

    reward_time_stamp = np.array(0)
    RL_DATA = {'action_fn': object,
               'rl_model': object,
               'enviro_fn': object,
               'reward_fn': object,
               'act_list': list,
               'max_epoch': int,
               'max_iter': int,
               'replay_buffer_max_sz': int,
               'replay_sz': int,
               'replay_trial': int,
               'replay_opt': object,
               'gamma': float,
               'alpha': float,
               'agent_update_interval': int,
               't_update_interval': int,
               't_update_rate': float,
               'qn': list,
               'tqn': list,
               'agent': openNeural,
               'SA_merge': bool,
               'epsilon': float,
               'epsilon_min': float,
               'epsilon_decay_fn': object,
               'epsilon_decay_rate': float,
               'epsilon_decay_threshold': float,
               'epsilon_decay_threshold_rate': float,
               'total_reward': float,
               'epoch': int}

    def __init__(self):
        pass

    def RL_SETTING(self,

                   rl_model, enviro_fn, reward_fn, act_list,
                   max_epoch, max_iter,
                   replay_buffer_max_sz, replay_sz, replay_trial, replay_opt,
                   gamma, alpha, agent_update_interval, t_update_interval, t_update_rate,
                   SA_merge, epsilon_decay_fn = None, action_fn = RL_DIRECT_ACTION):
        # -------RL setting-----------
        self.RL_DATA["action_fn"] = action_fn
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
        self.RL_DATA["replay_opt"] = replay_opt
        # ---------learning-----------
        self.RL_DATA["gamma"] = gamma
        self.RL_DATA["alpha"] = alpha
        self.RL_DATA["agent_update_interval"] = agent_update_interval
        self.RL_DATA["t_update_interval"] = t_update_interval
        self.RL_DATA["t_update_rate"] = t_update_rate
        self.RL_DATA["qn"] = np.empty(0)
        self.RL_DATA["tqn"] = np.empty(0)
        self.RL_DATA["epsilon"] = 0
        self.RL_DATA["epsilon_decay_fn"] = epsilon_decay_fn
        self.RL_DATA["SA_merge"] = SA_merge

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
        q_network.xavier_initialization()
        q_network.opt_reset()
        q_network.learning_set(learning_rate=learning_rate,
                               dropout_rate=dropout_rate,
                               loss_fun=loss_fun,
                               learn_optima=learn_optima)
        # target neural network
        tq_network = deepcopy(q_network)
        self.RL_DATA["qn"] = np.append(self.RL_DATA["qn"], q_network)
        self.RL_DATA["tqn"] = np.append(self.RL_DATA["tqn"], tq_network)
        self.RL_DATA["agent"] = self.RL_DATA["qn"][0]

    def E_G_DECAY_SETTING(self,
                          initial_epsilon=1,
                          epsilon_decay_rate=0.9,
                          decay_threshold=1,
                          decay_threshold_rate=0.8,
                          decay_minimum=0.1,
                          decay_fn=E_G_DECAY_BY_REWARD):
        self.RL_DATA["epsilon"] = initial_epsilon
        self.RL_DATA["epsilon_min"] = decay_minimum
        if decay_fn is None:
            self.RL_DATA["epsilon_decay_fn"] = E_G_DECAY_BY_REWARD
        else:
            self.RL_DATA["epsilon_decay_fn"] = decay_fn
        self.RL_DATA["epsilon_decay_rate"] = epsilon_decay_rate
        self.RL_DATA["epsilon_decay_threshold"] = decay_threshold
        self.RL_DATA["epsilon_decay_threshold_rate"] = decay_threshold_rate

    def RL_RUN(self, initial_state, terminate_reward_condition=None):
        # ----------playing initialization----------
        __update = 0
        replay_buffer = np.empty((0, 5))
        update_step = 0
        # ----------playing episode----------
        for __ep in range(self.RL_DATA["max_epoch"]):
            # ----------playing----------
            self.RL_DATA["total_reward"] = 0
            self.RL_DATA["epoch"] = __ep
            s = initial_state.copy()  # initialize state S
            for __it in range(self.RL_DATA["max_iter"]):  # increase the interation
                update_step += 1
                # ----------Get exp from action----------
                r, t = self.RL_DATA["reward_fn"](s)  # get [r, t] (reward and terminated state)
                a = self.RL_DATA["action_fn"](s=s, rl_data_dict=self.RL_DATA)  # get a (action)
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
                self.RL_DATA["total_reward"] += r
                # ---------Info -----------------
                if self.RL_DATA["epsilon_decay_fn"] is not None:
                    self.RL_DATA["epsilon_decay_fn"](self.RL_DATA)
                # ---------Info------------------
                print(Fore.RESET, __ep, __it, Fore.LIGHTCYAN_EX, s, Fore.LIGHTYELLOW_EX, a, Fore.LIGHTYELLOW_EX, r,
                      Fore.LIGHTCYAN_EX, ss, t, Fore.LIGHTGREEN_EX, self.RL_DATA["epsilon"])
                # ---------Termination check -----------------
                if t:
                    break
                s = ss.copy()  # update new state
            # ---------graph-----------------
            self.reward_time_stamp = np.append(self.reward_time_stamp, self.RL_DATA["total_reward"])
            if __it == self.RL_DATA["max_epoch"]:
                break
            if terminate_reward_condition is not None and self.RL_DATA["total_reward"] >= terminate_reward_condition:
                break

    def RL_SAVE(self, file_name) -> None:
        os.makedirs("save", exist_ok=True)
        setting_file = np.array([
            self.RL_DATA['action_fn'],
            self.RL_DATA[ 'rl_model'],
            self.RL_DATA['enviro_fn'],
            self.RL_DATA['reward_fn'],
            self.RL_DATA['act_list'],
            self.RL_DATA['max_epoch'],
            self.RL_DATA['max_iter'],
            self.RL_DATA['replay_buffer_max_sz'],
            self.RL_DATA['replay_sz'],
            self.RL_DATA['replay_trial'],
            self.RL_DATA['replay_opt'],
            self.RL_DATA['gamma'],
            self.RL_DATA['alpha'],
            self.RL_DATA['agent_update_interval'],
            self.RL_DATA['t_update_interval'],
            self.RL_DATA['t_update_rate'],
            self.RL_DATA['qn'],
            self.RL_DATA['tqn'],
            self.RL_DATA['agent'],
            self.RL_DATA['SA_merge'],
            self.RL_DATA['epsilon'],
            self.RL_DATA['epsilon_min'],
            self.RL_DATA['epsilon_decay_fn'],
            self.RL_DATA['epsilon_decay_rate'],
            self.RL_DATA['epsilon_decay_threshold'],
            self.RL_DATA['epsilon_decay_threshold_rate'],
            self.RL_DATA['total_reward'],
            self.RL_DATA['epoch'],
            self.RL_DATA["qn"],
            self.RL_DATA["tqn"]], dtype=object)
        np.save("save/" + file_name + "_info", setting_file, allow_pickle=True)
        print("SAVE FINISH")

    def RL_LOAD(self, file_name) -> None:
        [self.RL_DATA['action_fn'],
            self.RL_DATA[ 'rl_model'],
            self.RL_DATA['enviro_fn'],
            self.RL_DATA['reward_fn'],
            self.RL_DATA['act_list'],
            self.RL_DATA['max_epoch'],
            self.RL_DATA['max_iter'],
            self.RL_DATA['replay_buffer_max_sz'],
            self.RL_DATA['replay_sz'],
            self.RL_DATA['replay_trial'],
            self.RL_DATA['replay_opt'],
            self.RL_DATA['gamma'],
            self.RL_DATA['alpha'],
            self.RL_DATA['agent_update_interval'],
            self.RL_DATA['t_update_interval'],
            self.RL_DATA['t_update_rate'],
            self.RL_DATA['qn'],
            self.RL_DATA['tqn'],
            self.RL_DATA['agent'],
            self.RL_DATA['SA_merge'],
            self.RL_DATA['epsilon'],
            self.RL_DATA['epsilon_min'],
            self.RL_DATA['epsilon_decay_fn'],
            self.RL_DATA['epsilon_decay_rate'],
            self.RL_DATA['epsilon_decay_threshold'],
            self.RL_DATA['epsilon_decay_threshold_rate'],
            self.RL_DATA['total_reward'],
            self.RL_DATA['epoch'],
            self.RL_DATA["qn"],
            self.RL_DATA["tqn"]] = np.load("save/" + file_name + "_info.npy", allow_pickle=True)
        print("LOAD FINISH")
