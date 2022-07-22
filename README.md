# OpenNeural
GNU license 2022(c) Useop Gim\
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
    B = openNeural()
    B.add_layer(4, 'x')
    B.add_layer(10, 'softReLU')
    B.add_layer(4, 'x')
    B.generate_weight()
    B.xavier_initialization()
    B.learn_set([1,2,3,4],[4,3,2,1], dropout = 0, optima = False)
    B.learn_start(max_trial= 100000, show_result=True)
```
