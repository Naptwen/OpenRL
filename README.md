# OpenNeural
GNU license 2022(c) Useop Gim\
v1.5.0\
Change algorithm order for SGD and others\
This version I tested for DDQN and DQN and it success! \
I am so happy that I finally make it for RL!\
*The RL version will be publish as soon as possible after some additional work for optimization \
**paraller and GPGPU will be publised as soon as after add some function for mutiple network parallel algorith

v1.4.0\
Change the variable name of algorithm\
Fix forget Load and Save csv for Bias layer\

v1.3.0\
-Change the matrix form of the neural network algorithm\
-Remove OpenCL algorithm for future work\
If you intereseted in some of my work please visit below or send an email\
https://sites.google.com/view/gesope/projects/a-i/reinforcement-neural-network-python?authuser=0

```python
import OpenNeural
if __name__ == '__main__':
# this is one iteration if you want to epoch do while and check the error for data
    B = openNeural()
    B.add_layer(4, 'x')
    B.add_layer(10, 'ReLU')
    B.add_layer(4, 'x')
    B.generate_weight()
    B.xavier_initialization()
    B.learning_reset()
    B.learning_set()
    start = time.time()
    for i in range(1000):
        B.learn_start(input_val=[1,2,3,4], target_val=[4,3,2,1])
        if B.error <= 0.01:
            break
        print(B.error)
    print('Hello NEURAL : ', time.time() - start)
```
