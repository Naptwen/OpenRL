# OpenNeural and OpenRL
GNU AFFERO GPL (c) Useop Gim 2022

OpenRL
v1.1.0
Had edit the variable and reorder the algorithm for a more user friendly

OpenRL
v1.0.1
- New : Finally add some success off policy RL algorithm and others\
Below is test code

OpenNeural
v1.7.1
- Fix : Huber loss function
- Edit :  Load Save for all layer
- New : Return Weight and Bias update record layer (for future parallel)
- New : NADAM (Nestrov + Adam)
- New : Pseudo Huber

OpenNeural
v1.7.0\
Change some normlization code (softmax and cross product)\
- New function : KLD JSD shannon_entropy\
- Edit function : leakReLU (0.3->0.03), all log base probabilty function change to log(x+1) to prevent inf nan error\
- Fix function : parametric function (a is user depend value), gradient for softmax and cross entropy function \
- Change function : the order of copy X layer by Normlizae, now overlapping X layer. So the back propagation order is changed also\
                    Return directly output value from run function for reducing writing the code length.\
- About RL : I coded algorithm for DQN, DDQN, D2QN, D3QN, and SAC and it works for those.\
~~Some find : For SAC, I increase the performance for the RL neural network by increasing difficulty of the level of game and it looks like work~~

OpenNeural
v1.6.1\
Add explanations for each function

OpenNeural
v1.6.0\
New option feature : normlization between each layer \
New function : he initialization\
New variable type : using function pointer\
Remove function : instead using run_int and cpu_run, merge it as run function\
Change function : change the comments for function explanation\
ETC : to avoid variable distortion, using private
**Now just focus on off policy RL 

OpenNeural
v1.5.0\
Change algorithm order for SGD and others\
This version I tested for DDQN and DQN and it success! \
I am so happy that I finally make it for RL!\
*The RL version will be publish as soon as possible after some additional work for optimization \
**paraller and GPGPU will be publised as soon as after add some function for mutiple network parallel algorith

OpenNeural
v1.4.0\
Change the variable name of algorithm\
Fix forget Load and Save csv for Bias layer

OpenNeural
v1.3.0\
-Change the matrix form of the neural network algorithm\
-Remove OpenCL algorithm for future work\
If you intereseted in some of my work please visit below or send an email\
https://sites.google.com/view/gesope/projects/a-i/reinforcement-neural-network-python?authuser=0

DNN version
```python
from OpenNeural import *
if __name__ == '__main__':
# this is one iteration if you want to epoch do while and check the error for data
    B = openNeural()
    B.add_layer(4)
    B.add_layer(10, ReLU, znormal)
    B.add_layer(4, parametricReLU(a = 0.03), znormal)
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

How Apply it?\
Example for Basic strucutre of LSTM\
The structure is refernced from https://en.wikipedia.org/wiki/Long_short-term_memory#:~:text=A%20common%20LSTM%20unit%20is%20composed%20of%20a,of%20information%20into%20and%20out%20of%20the%20cell.
```python
from OpenNeural import *
class LSTM:
   def __init__(self, input_sz, hidden_sz):
        self.forget_gate = openNeural()
        self.forget_gate.add_layer(input_sz)
        self.forget_gate.add_layer(hidden_sz, ReLU, sigmoid)
        self.forget_gate.add_layer(1)
        self.forget_gate.generate_weight()
        self.forget_gate.he_initialization()
        self.forget_gate.opt_reset()
        self.forget_gate.learning_set()

        self.input_gate = openNeural()
        self.input_gate.add_layer(input_sz)
        self.input_gate.add_layer(hidden_sz, ReLU, sigmoid)
        self.input_gate.add_layer(1)
        self.input_gate.generate_weight()
        self.input_gate.he_initialization()
        self.input_gate.opt_reset()
        self.input_gate.learning_set()

        self.tanh_gate = openNeural()
        self.tanh_gate.add_layer(input_sz)
        self.tanh_gate.add_layer(hidden_sz, ReLU, arctan)
        self.tanh_gate.add_layer(1)
        self.tanh_gate.generate_weight()
        self.tanh_gate.he_initialization()
        self.tanh_gate.opt_reset()
        self.tanh_gate.learning_set()

        self.out_gate = openNeural()
        self.out_gate.add_layer(input_sz)
        self.out_gate.add_layer(hidden_sz, ReLU, sigmoid)
        self.out_gate.add_layer(1)
        self.out_gate.generate_weight()
        self.out_gate.he_initialization()
        self.out_gate.opt_reset()
        self.out_gate.learning_set()


    def run_gate(self, input_val, past_c, past_h):
        assert len(input_val) + len(past_h) <= len(self.forget_gate.get_layer()[0])
        if past_h is None:
            input_list = np.zeros(self.forget_gate.get_layer()[0])
            input_list[0:len(input_val)] = input_val
        else:
            input_list = np.apend(input_val, past_h)
        self.forget_gate.run(input_list)
        self.input_gate.run(input_list)
        self.tanh_gate.run(input_list)
        self.out_gate.run(input_list)
        C = self.forget_gate.output * past_c.output \
            + self.input_gate.output * self.tanh_gate.output
        h = self.out_gate.output * arctan(arctan(C))
        return C, h
```
ReinForcement Exmaple
```py
import time
from matplotlib import pyplot as plt

import numpy as np
import pygame

from openRL import *
import random

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
        random_seed = int(time.time())
        self.neural = openRL(model=DDQN, action_size=3, random_seed=random_seed)
        self.neural.RL_LEARN_SETTING(
            enviro_fn=enviro, reward_fn=reward_policy, max_iter=120, max_epoch=800,
            buffer_maximum_sz=32, buffer_replay_sz=1, buffer_replay_trial=1,
            dropout_rate=0.0,
            learning_rate=0.005, learn_optima='NADAM', loss_fun=HUBER,
            w_update_interval=5, t_update_interval=10, gamma=0.99, t_update_rate=0.01
        )

        self.neural.RL_SETTING(q_layer=[4, 8, 12, 3],
                               q_activation_fn=[linear_x, leakReLU, leakReLU, linear_x],
                               q_normalization=[linear_x, znormal, znormal, linear_x])
        self.neural.RL_LOAD(file_name='TEST__')

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

    def __init__(self, random_seed):
        self.neural = openRL(D3QN, 3, random_seed)
        self.neural.RL_LOAD('TEST__')
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
            act = RL_ACTION(s=np.array([self.cx, self.cy, self.ball_x, self.ball_y]), epsilon=0.0, rl_data_dict=self.neural.RL_DATA)
            print(
                Fore.LIGHTRED_EX,
                f'Ac : {act}',
                Fore.LIGHTBLACK_EX,
                [self.cx, self.cy, self.ball_x, self.ball_y],
                Fore.LIGHTBLUE_EX,
                f'{self.neural.RL_DATA["qn"].run(np.array([self.cx, self.cy, self.ball_x, self.ball_y]))}',
                Fore.LIGHTYELLOW_EX,
                f'qn : {np.where(self.neural.RL_DATA["qn"].output[0:3] == np.max(self.neural.RL_DATA["qn"].output[0:3]), 1, 0)}',
                Fore.LIGHTBLUE_EX,
                total_reward,
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
    A.neural.RL_SAVE('TEST__')  
    B = PY_GAME(int(time.time()))  

```
