# OpenNeural
GNU AFFERO GPL (c) Useop Gim 2022

v1.6.2\
Add bias update for RNN type

v1.6.1\
New function :  generate_rnn_set

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
import OpenNeural
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

RNN version

```python
import OpenNeural
    B = openNeural()
    B.add_layer(4)
    B.add_layer(10, ReLU, linear_x) # it automatically be the same as next layer but the shape should be the same next layer
    B.add_layer(10, ReLU, znormal)
    B.add_layer(4)
    B.generate_weight()
    B.xavier_initialization()
    B.generate_rnn_set()  # rnn setting
    B.opt_reset()
    B.learning_set()
    B.show_layer()
    start = time.time()
    for i in range(1000):
        B.run(input_val=np.array([1, 2, 3, 4]))
        B.learn_start(out_val=B.output, target_val=np.array([4, 3, 2, 1]))
        if B.error <= 0.01:
            break
        print(B.error)
    print('Hello NEURAL : ', time.time() - start, B.output)
```
