# OpenNeural
GNU AFFERO GPL (c) Useop Gim 2022\
v1.7.0\
Change some normlization code (softmax and cross product)\
New function : KLD JSD shannon_entropy\
Edit function : leakReLU (0.3->0.03), all log base probabilty function change to log(x+1) to prevent inf nan error\
Fix function : parametric function (a is user depend value), gradient for softmax and cross entropy function \
Change function : the order of copy X layer by Normlizae, now overlapping X layer. So the back propagation order is changed also\
Return directly output value from run function for reducing writing the code length.\
About RL : I coded algorithm for DQN, DDQN, D2QN, D3QN, and SAC and it works for those.\
Some find : For SAC, I increase the performance for the RL neural network by increasing difficulty of the level of game and it looks like work.

v1.6.1\
Add explanations for each function

v1.6.0\
New option feature : normlization between each layer \
New function : he initialization\
New variable type : using function pointer\
Remove function : instead using run_int and cpu_run, merge it as run function\
Change function : change the comments for function explanation\
ETC : to avoid variable distortion, using private
**Now just focus on off policy RL 

v1.5.0\
Change algorithm order for SGD and others\
This version I tested for DDQN and DQN and it success! \
I am so happy that I finally make it for RL!\
*The RL version will be publish as soon as possible after some additional work for optimization \
**paraller and GPGPU will be publised as soon as after add some function for mutiple network parallel algorith

v1.4.0\
Change the variable name of algorithm\
Fix forget Load and Save csv for Bias layer

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
    B.add_layer(4, parametricReLU, znormal)
    B.add_layer(4)
    B.generate_weight()
    B.xavier_initialization()
    B.opt_reset()
    B.learning_set()
    start = time.time()
    for i in range(1000):
        B.run(input_val = np.array([1,2,3,4]))
        B.learn_start(out_val = B.output, target_val=np.array([4,3,2,1]))
        if B.error <= 0.01:
            break
        print(B.error)
    print('Hello DNN NEURAL : ', time.time() - start, B.output)
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
