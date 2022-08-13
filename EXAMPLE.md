GNU AFFERO (C) 2022 Useop Gim.
C++ SAC version
```cpp
#pragma once
#include "usgRL.hpp"
#include <windows.h>

float reward_function(const vector<float>& s)
{
	if (s[0] == s[2] && s[1] == s[3]) return 1;
	else return -sqrtf(((s[0] - s[2]) * (s[0] - s[2]) + (s[1] - s[3]) * (s[1] - s[3])));
}

bool enviro_function(const vector<float>& s, vector<float>& ss, vector<float>& act_list, long long a)
{
	uniform_int_distribution<int> dis(0, 4);
	switch (a)
	{
	case 1 : 
		ss[0] -= 1; break;
	case 2 :
		ss[0] += 0; break;
	case 3:
		ss[0] += 1; break;
	case 4:
		ss[1] += 1; break;
	case 5:
		ss[1] -= 1; break;
	default:
		ss[0] += 0;
	}
	if (ss[0] < 0 || ss[0] > 4)
	{
		ss[0] = 0;
		ss[1] = 0;
		return true;
	}
	if (ss[1] < 0 || ss[1] > 4)
	{
		ss[0] = 0;
		ss[1] = 0;
		return true;
	}
	else if (ss[0] == ss[2] && ss[1] == ss[3])
	{
		ss[2] = dis(rngmachine);
		ss[3] = dis(rngmachine);
		if (ss[2] == 0 && ss[3] == 0)
		{
			uniform_int_distribution<int> dis2(1, 4);
			ss[2] = dis2(rngmachine);
			ss[3] = dis2(rngmachine);
		}
		return false;
	}
	return false;
}

void draw(const vector<float>& s)
{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
				if (i == s[0] && j == s[1])
					printf("[x]");
				else if (i == s[2] && j == s[3])
					printf("[G]");
				else
					printf("[ ]");
			printf("\n");
		}
}

int main()
{
	auto A = openNeural<float>();
	A.add_layer(4, linear_x, linear_x);
	A.add_layer(16, ReLU, linear_x);
	A.add_layer(8, ReLU, linear_x);
	A.add_layer(5, softmax, znormal);
	A.generate_layer();
	A.opt_reset();
	A.xavier_init();
	A.learning_set(KL_DIVERGENCE, 0,0.03f, 0.0, 100, ADAM);

	auto B = openNeural<float>();
	B.add_layer(5, linear_x, linear_x);
	B.add_layer(36, leakReLU, znormal);
	B.add_layer(1, linear_x, linear_x);
	B.generate_layer();
	B.opt_reset();
	B.xavier_init();
	B.learning_set(MSE2, 0, 0.03f, 0.0, 100, NADAM);

	auto C = openNeural<float>();
	C.add_layer(5, linear_x, linear_x);
	C.add_layer(36, leakReLU, znormal);
	C.add_layer(1, linear_x, linear_x);
	C.generate_layer();
	C.opt_reset();
	C.xavier_init();
	C.learning_set(MSE2, 0, 0.03f, 0.0, 100, NADAM);


	auto R = openSAC<float>();
	R.RL_ADD_AGENT(A);
	R.RL_ADD_AGENT(B);
	R.RL_ADD_AGENT(C);
	R.RL_REPLAY_SETTING(32, 4, 4, 4, 0.01f);
	R.RL_PLAY_SETTING(enviro_function, reward_function,  { 1,2,3,4,5 });
	vector<float> s = { 0,0, 3,3 };
	vector<float> ss = { 0,0, 3, 3 };
	vector<string> action_list = { "up","    stop","  down","  left","  right" };
	int iter = 0;
	while (iter++ < 20000)
	{
		system("cls");
		EXP<float> exp = R.RL_PLAY(s);
	}
	R.__agents[0].file_save("actor.txt");
	R.__agents[1].file_save("critic.txt");
	return 0;
}
```


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
DDQN {S}
```py
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
            rl_model=DDQN,
            enviro_fn=enviro,
            reward_fn=reward_policy,
            act_list=[0, 1, 2],
            max_epoch=200,
            max_iter=300,
            replay_buffer_max_sz=32,
            replay_sz=1,
            replay_trial=1,
            replay_opt=REPLAY_PRIORITIZATION,
            gamma=0.99,
            alpha=0.001,
            agent_update_interval=5,
            t_update_interval=10,
            t_update_rate=0.01,
            epsilon_decay_fn=E_G_DECAY_BY_REWARD,
            sa_merge=False)
        self.neural.CREATE_Q(learning_rate=0.001, dropout_rate=0.0, loss_fun=HUBER, learn_optima='NADAM',
                             q_layer=np.array([4, 8, 12, 3]),
                             q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                             q_normalization=np.array([linear_x, znormal, znormal, softmax], dtype=object))
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
        self.neural.RL_DATA["epsilon"] = 0
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
            act = RL_ACTION(s=[self.cx, self.cy, self.ball_x, self.ball_y], rl_data_dict=self.neural.RL_DATA)
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

D3QN {S,A}
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
            rl_model=D3QN,
            enviro_fn=enviro,
            reward_fn=reward_policy,
            act_list=[0, 1, 2],
            max_epoch=200,
            max_iter=300,
            replay_buffer_max_sz=32,
            replay_sz=1,
            replay_trial=1,
            replay_opt=REPLAY_PRIORITIZATION,
            gamma=0.99,
            alpha=0.001,
            agent_update_interval=5,
            t_update_interval=10,
            t_update_rate=0.01,
            epsilon_decay_fn=E_G_DECAY_BY_REWARD,
            sa_merge=True)
        self.neural.CREATE_Q(learning_rate=0.001, dropout_rate=0.0, loss_fun=HUBER, learn_optima='NADAM',
                             q_layer=np.array([5, 8, 12, 2]),
                             q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                             q_normalization=np.array([linear_x, znormal, znormal, softmax], dtype=object))
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
        self.neural.RL_DATA["epsilon"] = 0
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
            act = RL_ACTION(s=[self.cx, self.cy, self.ball_x, self.ball_y], rl_data_dict=self.neural.RL_DATA)
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
SAC
```python
import time
from matplotlib import pyplot as plt
import pygame
from openRL import *
import cProfile


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
            rl_model=SAC,
            enviro_fn=enviro,
            reward_fn=reward_policy,
            act_list=[0, 1, 2],
            max_epoch=200,
            max_iter=300,
            replay_buffer_max_sz=32,
            replay_sz=1,
            replay_trial=1,
            replay_opt=None,
            gamma=0.99,
            alpha=0.001,
            agent_update_interval=1,
            t_update_interval=10,
            t_update_rate=0.01,
            epsilon_decay_fn=None,
            sa_merge=False)
        self.neural.CREATE_Q(learning_rate=0.001, dropout_rate=0.0, loss_fun=HUBER, learn_optima='NADAM',
                             q_layer=np.array([4, 8, 12, 3]),
                             q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                             q_normalization=np.array([linear_x, znormal, znormal, softmax], dtype=object))
        self.neural.CREATE_Q(learning_rate=0.001, dropout_rate=0.0, loss_fun=HUBER, learn_optima='NADAM',
                             q_layer=np.array([5, 8, 12, 1]),
                             q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                             q_normalization=np.array([linear_x, znormal, znormal, linear_x], dtype=object))
        self.neural.CREATE_Q(learning_rate=0.001, dropout_rate=0.0, loss_fun=HUBER, learn_optima='NADAM',
                             q_layer=np.array([5, 8, 12, 1]),
                             q_activation_fn=np.array([linear_x, leakReLU, leakReLU, linear_x], dtype=object),
                             q_normalization=np.array([linear_x, znormal, znormal, linear_x], dtype=object))
        self.neural.RL_LOAD('Test__')
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
        self.neural.RL_DATA["epsilon"] = 0
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
            act = RL_ACTION(s=[self.cx, self.cy, self.ball_x, self.ball_y], rl_data_dict=self.neural.RL_DATA)
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
