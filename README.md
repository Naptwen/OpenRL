# OpenNeural and OpenRL
GNU AFFERO GPL (c) Useop Gim 2022\

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNaptwen%2FOpen_pyAI&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Open_pyAI v2.2.3
- Fix : Some index consution in the RL algorithm
- Nww : Reparameterization function and gradient descent for policy entworks

Open_pyAI v2.2.1
 - Edit: parameters of functions
 - New : merge RL_ACTION functino
 - Working : SAC policy gradient update and alpha automatically function with guassian distribution algorithm

**_NOTE_** there can be parallel algorithm in Q(S,A), neural network, mini-batch, prioritization replay buffer, A3C -- To sum up: there are several technics for applying parellizing; in the calculation algorithm of the matrix, finding the Q value by the state with all actions, reinforcement buffer run algorithm and gradient calculation for Y value and Q value, and buffer seperating then merging. 

Open_pyAI v2.2.1
 - Edit: changed the algorithm structure for be more general
 - Fix: some reinforcement algorithms fix

OpenRL
v2.0.0
 - Edit : Seperate class and functions for reinforcement learning, setting functions, load and save functions
 - Edit : editing reinforcement learning model functions
 - New : function by policy setting SAC (not fully compelted)

v1.2.1
- Since there are two type of calculating Q value; "[s,ai]->Q(s,ai) for i in Z" and "[s]->Q(s,a1,a2,a3....)"
- Thus edited algorithm for working in both way by set the variable of using act_sz and act_list
- New : new function and variable for input action and status together for Q value reinforcecment learning
- Fix : fix some save load algorithm

OpenRL
v1.1.0 and OpenNeural v1.7.2
- Had edit the variable and reorder the algorithm for a more user friendly

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
v1.7.0
-Change some normlization code (softmax and cross product)
- New function : KLD JSD shannon_entropy
- Edit function : leakReLU (0.3->0.03), all log base probabilty function change to log(x+1) to prevent inf nan error
- Fix function : parametric function (a is user depend value), gradient for softmax and cross entropy function 
- Change function : the order of copy X layer by Normlizae, now overlapping X layer. So the back propagation order is changed also\
                    Return directly output value from run function for reducing writing the code length.
- About RL : I coded algorithm for DQN, DDQN, D2QN, D3QN, and SAC and it works for those.
~~Some find : For SAC, I increase the performance for the RL neural network by increasing difficulty of the level of game and it looks like work~~

OpenNeural
v1.6.1\
- Add explanations for each function

OpenNeural
v1.6.0\
- New option feature : normlization between each layer 
- New function : he initialization
- New variable type : using function pointer
- Remove function : instead using run_int and cpu_run, merge it as run function
- Change function : change the comments for function explanation
- ETC : to avoid variable distortion, using private
**Now just focus on off policy RL 

OpenNeural
v1.5.0\
- Change algorithm order for SGD and others
- This version I tested for DDQN and DQN and it success! 
- I am so happy that I finally make it for RL!
*The RL version will be publish as soon as possible after some additional work for optimization 
**paraller and GPGPU will be publised as soon as after add some function for mutiple network parallel algorith

OpenNeural
v1.4.0\
- Change the variable name of algorithm
- Fix forget Load and Save csv for Bias layer

OpenNeural
v1.3.0\
- Change the matrix form of the neural network algorithm
- Remove OpenCL algorithm for future work
- If you intereseted in some of my work please visit below or send an email
- https://sites.google.com/view/gesope/projects/a-i/reinforcement-neural-network-python?authuser=0
