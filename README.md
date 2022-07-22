# OpenNeural
GNU license 2022(c) Useop Gim\
v1.5.0\
Change algorithm order for SGD and others\
This version I tested for DDQN and DQN and it success! I am so impressived my working actually be used for RL

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
    B.add_layer(10, 'softReLU')
    B.add_layer(4, 'x')
    B.generate_weight()
    B.xavier_initialization()
    B.learn_start(max_trial= 100000, show_result=True)
    B.learning_reset()
    B.learning_set(learning_rate=learning_rate, dropout_rate=dropout_rate, loss_fun=loss_fun, adam_rmsp=adam_rmsp, Error_optimaization=Error_optimaization)
    print(B.errror)
```
