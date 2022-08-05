

DNN version
```python
from OpenNeural import *
if __name__ == '__main__':
# this is one iteration if you want to epoch do while and check the error for data
    B = openNeural()
    B.add_layer(4)
    B.add_layer(10, ReLU, linear_x)
    B.add_layer(4, ReLU, linear_x)
    B.add_layer(4)
    B.generate_weight()
    B.xavier_initialization()
    B.opt_reset()
    B.learning_set()
    start = time.time()
    for i in range(1000):
        B.learn_start(out_val = B.run(input_val = np.array([1,2,3,4])), target_val=np.array([4,3,2,1]))
        if B.error <= 0.01:
            break
        print(B.error)
    print('Hello DNN : ', time.time() - start, B.output)
```
DDQN
```python
import time
from matplotlib import pyplot as plt
import pygame
from openRL import *

def reward_policy(s, g=None) -> [float, bool]:
    """
    Args:
        s(np.ndarray): current status
        g(np.ndarray): goal for get reward

    Return:
         Reward, Termination
    """
    # -------------REWARD----------------------
    if g is None:
        g = [s[0], s[1]]
        if s[2] == g[0] and s[3] == 4:
            r = 1
            t = False
        elif s[2] != g[0] and s[3] == 4:
            r = -1
            t = True
        else:
            r = 0
            t = False
        return r, t
    else:
        if (s[2] == g[0] and s[3] == g[1]) or (s[2] == s[0] and s[3] == s[1]):
            r = 1
            t = False
        elif s[2] != g[0] and s[3] == g[1]:
            r = -1
            t = True
        else:
            r = 0
            t = False
        return r, t

def enviro(s, a) -> np.ndarray:
    assert 0 <= a <= 3
    ss = s.copy()
    if a == 0:
        ss[0] -= 1
    elif a == 1:
        ss[0] += 0
    elif a == 2:
        ss[0] += 1
    if ss[0] < 0:
        ss[0] = 0
    elif ss[0] > 3:
        ss[0] = 3
    # ------------NEW STATE--------------------
    ss[3] += 1
    if ss[2] == ss[0] and ss[3] > 4:
        ss[2] = random.randint(0, 3)
        ss[3] = 0
    elif ss[2] != ss[0] and ss[3] > 4:
        ss[0] = 0
        ss[2] = random.randint(0, 3)
        ss[3] = 0
    return ss


class DQN_TEST():

    def start_Q_learning(self):
        self.neural = openRL()
        self.neural.RL_SETTING(
            action_fn=RL_E_G_ACTION,
            rl_model=DDQN,
            enviro_fn=enviro,
            reward_fn=reward_policy,
            act_list=[0, 1, 2],
            max_epoch=5000,
            max_iter=300,
            replay_buffer_max_sz=64,
            replay_sz=4,
            replay_trial=1,
            replay_opt=REPLAY_PRIORITIZATION,
            gamma=0.99,
            alpha=0.001,
            agent_update_interval=5,
            t_update_interval=10,
            t_update_rate=0.01,
            epsilon_decay_fn = E_G_DECAY_BY_REWARD,
            SA_merge = False) 
        self.neural.CREATE_Q(
            learning_rate=0.001,
            dropout_rate=0.0,
            loss_fun=HUBER,
            learn_optima='NADAM',
            q_layer=np.array([4, 8, 12, 3]),
            q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
            q_normalization=np.array([linear_x, znormal, znormal, linear_x], dtype=object))
        self.neural.E_G_DECAY_SETTING()
        self.neural.RL_RUN(initial_state=np.array([0, 4, random.randint(0, 3), 0]),
                           terminate_reward_condition=20)
        self.static_function()

    def static_function(self):
        xx = np.arange(len(self.neural.reward_time_stamp))
        yy = self.neural.reward_time_stamp
        plt.plot(xx, yy, 'k+', alpha=0.3)
        plt.title('Test graph')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.show()
        print(Fore.LIGHTCYAN_EX, 'FINISH')


class PY_GAME():
    cell = np.zeros([4, 5])
    cell_w = 50
    neural = object
    ball_x, ball_y, cx, cy = random.randint(0, 3), 0, 0, 4

    def __init__(self):
        self.neural = openRL()
        self.neural.RL_LOAD('Test__')
        self.test_play()

    def drawing(self):
        self.screen.fill([0, 0, 0])
        for i in range(self.cell.shape[0]):
            pygame.draw.line(self.screen, [255, 255, 255], [i * self.cell_w, 0],
                             [i * self.cell_w, self.cell.shape[1] * self.cell_w])
        for j in range(self.cell.shape[1]):
            pygame.draw.line(self.screen, [255, 255, 255], [0, self.cell_w * j],
                             [self.cell.shape[0] * self.cell_w, self.cell_w * j])
        for j in range(self.cell.shape[1]):
            for i in range(self.cell.shape[0]):
                if self.cell[i, j] == 1:
                    pygame.draw.rect(self.screen, [0, 0, 255],
                                     [i * self.cell_w, j * self.cell_w, self.cell_w, self.cell_w])
                elif self.cell[i, j] == 2:
                    pygame.draw.rect(self.screen, [255, 0, 0],
                                     [i * self.cell_w, j * self.cell_w, self.cell_w, self.cell_w])

    def test_play(self):
        # -------------PY GAME SETTING-------------
        pygame.init()
        pygame.font.init()
        clock = pygame.time.Clock()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)  # if you want to use this module.
        self.screen = pygame.display.set_mode([self.cell.shape[0] * self.cell_w, self.cell.shape[1] * self.cell_w])
        pygame.display.set_caption("MY DQN GAME")
        # -------------REINFORCEMENT SETTING-------------
        self.neural.__epsilon = 0
        # -------------GAME SETTING-------------
        self.ball_x = random.randint(0, 3)
        self.ball_y = 0
        self.cx = 0
        self.cy = 4
        total_reward = 0
        FPS = 10
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # ------------NEW STATE--------------------
            self.cell = np.zeros(self.cell.shape)
            self.cell[self.ball_x, self.ball_y] = 1
            self.cell[self.cx, self.cy] = 2
            # ------------DRAWING STATE--------------------
            clock.tick(FPS)
            self.drawing()
            text = my_font.render(str(int(total_reward)), True, (0, 255, 0), (0, 0, 128))
            textRect = text.get_rect()
            textRect.center = (self.cell.shape[0] * self.cell_w / 2, self.cell.shape[1] * self.cell_w / 2)
            self.screen.blit(text, textRect)
            pygame.display.flip()
            # -------------REWARD----------------------
            reward, t = reward_policy(np.array([self.cx, self.cy, self.ball_x, self.ball_y]))
            # ------------ACTION-----------------------
            act = RL_DIRECT_ACTION(s=np.array([self.cx, self.cy, self.ball_x, self.ball_y]), rl_data_dict=self.neural.RL_DATA)
            print(
                Fore.LIGHTRED_EX,
                total_reward,
                Fore.LIGHTBLACK_EX,
                [self.cx, self.cy, self.ball_x, self.ball_y],
                Fore.LIGHTYELLOW_EX,
                f'agent : {self.neural.RL_DATA["agent"].output[0:len(self.neural.RL_DATA["act_list"])]}',
                Fore.LIGHTBLUE_EX,
                f'Ac : {act}',
                Fore.RESET)
            self.cx, self.cy, self.ball_x, self.ball_y = enviro([self.cx, self.cy, self.ball_x, self.ball_y], act)
            total_reward += reward
            if reward < 0:
                total_reward = 0
            # -------------DRAWING---------------------
            clock.tick(FPS)
            self.drawing()
            text = my_font.render(str(int(total_reward)), True, (0, 255, 0), (0, 0, 128))
            textRect = text.get_rect()
            textRect.center = (self.cell.shape[0] * self.cell_w / 2, self.cell.shape[1] * self.cell_w / 2)
            self.screen.blit(text, textRect)
            pygame.display.flip()


if __name__ == '__main__':
    A = DQN_TEST()
    start = time.time()
    A.start_Q_learning()
    print('take time', time.time() - start)
    A.neural.RL_SAVE('Test__')
    B = PY_GAME()

```
D3QN EXAMPLE
```python
from openRL_method import *
import os


# GNU AFFERO GPL (c) Useop Gim 2022
# Please let reference
# If you intereseted in my work visit in
# https://sites.google.com/view/gesope/projects/a-i/neural-network-algorithm-explanation?authuser=0

class openRL:
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
                   sa_merge, epsilon_decay_fn=None) -> None:
        """

        :param rl_model:
        :type rl_model: function
        :param enviro_fn:
        :type enviro_fn: function
        :param reward_fn:
        :type reward_fn: function
        :param act_list:
        :type act_list: list
        :param max_epoch:
        :type max_epoch: int
        :param max_iter:
        :type max_iter: int
        :param replay_buffer_max_sz:
        :type replay_buffer_max_sz: int
        :param replay_sz:
        :type replay_sz: int
        :param replay_trial:
        :type replay_trial: int
        :param replay_opt:
        :type replay_opt: function
        :param gamma:
        :type gamma: float
        :param alpha:
        :type alpha: float
        :param agent_update_interval:
        :type agent_update_interval: int
        :param t_update_interval:
        :type t_update_interval: int
        :param t_update_rate:
        :type t_update_rate: float
        :param sa_merge:
        :type sa_merge: bool
        :param epsilon_decay_fn:
        :type epsilon_decay_fn: function
        """
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
        self.RL_DATA["SA_merge"] = sa_merge

    def CREATE_Q(self,
                 learning_rate, dropout_rate, loss_fun, learn_optima,
                 q_layer=np.array([4, 8, 12, 3]),
                 q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                 q_normalization=np.array([linear_x, znormal, znormal, linear_x], dtype=object),
                 ) -> None:
        """

        :param learning_rate:
        :type learning_rate: float
        :param dropout_rate:
        :type dropout_rate: float
        :param loss_fun:
        :type loss_fun: function
        :param learn_optima:
        :type learn_optima: str
        :param q_layer:
        :type q_layer: list or np.ndarray
        :param q_activation_fn:
        :type q_activation_fn: list or np.ndarray
        :param q_normalization: list or np.ndarray
        :type q_normalization: list or np.ndarray
        """

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
        """

        :param initial_epsilon:
        :type initial_epsilon: float
        :param epsilon_decay_rate:
        :type epsilon_decay_rate: float
        :param decay_threshold:
        :type decay_threshold: float
        :param decay_threshold_rate:
        :type decay_threshold_rate: float
        :param decay_minimum:
        :type decay_minimum: float
        :param decay_fn:
        :type decay_fn: function
        """
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
        """

        :param initial_state:
        :type initial_state: list or np.ndarray
        :param terminate_reward_condition:
        :type terminate_reward_condition: float
        """
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
                a = RL_ACTION(s=s, rl_data_dict=self.RL_DATA)  # get a (action)
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

    def RL_SAVE(self, file_name: str) -> None:
        os.makedirs("save", exist_ok=True)
        setting_file = np.array([
            self.RL_DATA['action_fn'],
            self.RL_DATA['rl_model'],
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

    def RL_LOAD(self, file_name: str) -> None:
        [self.RL_DATA['action_fn'],
         self.RL_DATA['rl_model'],
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

```
