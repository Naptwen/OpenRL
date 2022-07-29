import random
import time
import sympy as sym
from sympy import sqrt, diff
import numpy as np
import os

# GNU AFFERO GPL (c) Useop Gim 2022 
# Please let reference 
# If you intereseted in my work visit in https://sites.google.com/view/gesope/projects/a-i/neural-network-algorithm-explanation?authuser=0

def min_max_normal(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values) + 0.00000001)  # pretend the divide by 0


def user_fn(values, gradient=False):
    return_values = np.zeors(len(values))
    if gradient:
        for i, val in enumerate(values):
            sym.Symbol('x')
            return_values[i] = sym.simplify(diff(val))
    else:
        for i, val in enumerate(values):
            sym.Symbol('x')
            return_values[i] = sym.simplify(val)
    return return_values


def linear_x(values, gradient=False):
    if gradient:
        return np.ones(len(values))
    return values


def softmax(values, gradient=False):
    if gradient:
        return values - values ** 2
    f = np.exp(values)
    return f / np.sum(f)


def softmax_normal(values):
    f = np.exp(values - np.max(values))
    return f / np.sum(f)


def arctan(values, gradient=False):
    if gradient:
        return 1 / (1 + values ** 2)
    return np.arctan(values)


def sigmoid(values, gradient=False):
    if gradient:
        return values * (1 - values)
    return 1 / (1 + np.exp(-values))


def ReLU(values, gradient=False):
    if gradient:
        return np.where(values > 0, 1, 0)
    return np.where(values > 0, values, 0)


def parametricReLU(values, gradient=False):
    if gradient:
        return np.where(values > 0, 1, -0.3)
    return np.where(values > -0.3 * values, values, -0.3 * values)


def leakReLU(values, gradient=False):
    if gradient:
        return np.where(values > 0, 1, 0.3)
    return np.where(values > 0.3 * values, values, 0.3 * values)


def znormal(values, gradient=False):
    if gradient:
        return (values - np.mean(values)) / (np.std(values) + 0.00000001)  # pretend the divide by 0
    else:
        return (np.std(values) + 0.00000001) * values + np.mean(values)


def max_min_limit(values):
    values = np.where(values < -2, -2, values)
    values = np.where(values > 2, 2, values)
    return values


def drop_Out(layer, drop_per):
    assert 0 <= drop_per < 1
    mask = np.random.uniform(0, 1, layer.shape[0]) > drop_per
    return mask * layer.shape[0] / (1.0 - drop_per)


# LOSS FUNCTIONS

def MPE(x, y):
    """
    Mean Absolute Percentage Error
    """
    a = np.empty(0)
    for i, j in zip(x, y):
        if j != 0 and i != j:
            a = np.append(a, (i - j) / (j * np.abs(i - j)))
        else:
            a = np.append(a, 0)
    error = 100 * np.mean(np.abs((y - x) / (y + 0.000000000001)))
    return a, error


def MSE(x, y):
    """
    Mean Square Error
    """
    return 2 * (x - y), np.mean((y - x) ** 2)


def RMSE(x, y):
    """
    Root Mean Squared Error
    """
    return -(y - x), np.sqrt(np.mean((y - x) ** 2))


def CROSS_ENTROPY(x, y):
    return y / x, -np.sum(y * np.log(x))


def BINARY_CROSS(x, y):
    """
    Binary Cross Entropy
    """
    return -(  # y * log(a)
            y * np.log(x)
            # +(1-y)
            + (np.ones(len(x)) - y)
            # *log(1-a)
            * np.log(np.ones(len(x)) - x)
    ), -sum(y * np.log(x))


def RELATIVE_ENTROPY(x, y):
    """
    Relative Entropy Error
    """
    return -y / x, np.sum(y * np.log(y / x))


class openNeural:
    """
    This is the neural network class by Useop Gim
    After declare class below functions must be

    * add_layer() : adding layer

    * generate_weight() : generating the weight after adding layer

    """
    __W_layer = np.empty(0, dtype=np.double)  # Weight Layer
    __B_layer = np.empty(0, dtype=np.double)  # Bias Layer
    __Z_layer = np.empty(0, dtype=np.double)  # Sum of weight times value Layer
    __X_layer = np.empty(0, dtype=np.double)  # Z + B Layer
    __N_layer = np.empty(0, dtype=object)  # Batch normal Layer
    __A_layer = np.empty(0, dtype=np.double)  # Activation Layer
    __EQ_layer = np.empty(0, dtype=object)  # Equation layer function pointer
    __Layer_shape = np.empty(0, dtype=np.int64)  # it contains Layer shapes information
    __VtW_layer = np.empty(0, dtype=np.double)  # Velocity layer by W for RMSP
    __MtW_layer = np.empty(0, dtype=np.double)  # Momentum layer by W for Adam
    __VtB_layer = np.empty(0, dtype=np.double)  # as same as above but by B
    __MtB_layer = np.empty(0, dtype=np.double)  # as same as above but by B
    __gE_layer = np.empty(0, dtype=np.double)  # It contains the error for each result and A layer
    __processor = str
    __drop_Out_rate = float  # drop out rate
    __learning_rate = float
    __learn_optima = 'ADAMRMSP'
    __loss_fun = object
    __iteration = int  # For using the AdamRMSP iteration value
    __beta_1 = float  # For using velocity rate
    __beta_2 = float  # For using momentum rate
    __epsilon = float  # For using velocity rate (to prevent dividing by 0)
    __RNN = False
    output = np.empty(0)
    target_val = np.empty(0, dtype=np.double)  # It is target value, same size of last value of the layer shape
    error = 1000  # loss function's error

    def __init__(self):
        self.__beta_1 = 0.9
        self.__beta_2 = 0.9
        self.__epsilon = 0.00000001

    def __cpu_run(self) -> np.ndarray:
        """
        This is cpu process forward
        """
        a_next = 0
        w_next = 0
        for i in range(len(self.__Layer_shape)):
            a_shape = self.__Layer_shape[i]
            # linearly sum
            self.__X_layer[a_next: a_next + a_shape] \
                = self.__Z_layer[a_next: a_next + a_shape] + self.__B_layer[a_next: a_next + a_shape]
            # normalization
            XN_array = self.__N_layer[i](self.__X_layer[a_next: a_next + a_shape])
            # activation function part
            self.__A_layer[a_next: a_next + a_shape] = self.__EQ_layer[i](XN_array)
            # checking drop out
            if self.__drop_Out_rate != 0 and i < len(self.__Layer_shape) - 1:
                self.__A_layer[a_next: a_next + a_shape] = self.drop_Out(self.__A_layer[a_next: a_next + a_shape],
                                                                         self.__drop_Out_rate)
            elif self.__drop_Out_rate == 1 and i < len(self.__Layer_shape) - 1:
                self.__A_layer[a_next: a_next + a_shape] = np.zeors(self.__Layer_shape[i])
            # obtained multiply weight
            if i < len(self.__Layer_shape) - 1:
                w_shape = self.__Layer_shape[i] * self.__Layer_shape[i + 1]
                self.__Z_layer[a_next + a_shape: a_next + a_shape + self.__Layer_shape[i + 1]] = \
                    np.matmul(self.__A_layer[a_next: a_next + a_shape],
                              self.__W_layer[w_next: w_next + w_shape].reshape(self.__Layer_shape[i],
                                                                               self.__Layer_shape[i + 1])).flatten()
                w_next += w_shape
            a_next += a_shape
        self.output = self.__A_layer[-self.__Layer_shape[-1]:]
        return self.output

    def __cpu_back(self) -> None:
        """
        This is cpu process back propagation
        """
        self.iteration += 1
        a_next = len(self.__A_layer) - self.__Layer_shape[-1]
        w_next = len(self.__W_layer) - self.__Layer_shape[-1] * self.__Layer_shape[-2]
        for i in reversed(range(1, len(self.__Layer_shape))):
            a_shape = self.__Layer_shape[i]
            w_shape = self.__Layer_shape[i] * self.__Layer_shape[i - 1]
            dE_dA = self.__gE_layer[a_next: a_next + a_shape]
            # batch normal recovery
            XN = self.__N_layer[i](self.__X_layer[a_next: a_next + a_shape], True)
            # calculate partial derivative of A
            dA_dX = self.__EQ_layer[i](XN, True)
            # dX_dZ = 1
            dE_dZ = dE_dA * dA_dX
            # converting dZ_dA for multiplication
            repeat_dZ_dA = np.repeat([self.__A_layer[a_next - self.__Layer_shape[i - 1]: a_next]], repeats=a_shape,
                                     axis=0)
            dig_dE_dZ = np.diag(dE_dZ)
            # gradient dE_dW
            dE_dW = np.matmul(dig_dE_dZ, repeat_dZ_dA).transpose().flatten()  # (dig * repeat)^T is [a,b][2a 2b][3a 3b]
            if self.__learn_optima == 'ADAMRMSP':
                if not self.__RNN or (self.__RNN and i % 2 == 1):  # if the past RNN
                    # AdamRMSP Weight
                    self.__MtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_1 * self.__MtW_layer[w_next:w_next + w_shape] + (1 - self.__beta_1) * dE_dW
                    self.__VtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_2 * self.__VtW_layer[w_next:w_next + w_shape] + (1 - self.__beta_2) * dE_dW ** 2
                    mdw_corr = self.__MtW_layer[w_next:w_next + w_shape] / (1 - self.__beta_1 ** self.iteration)
                    vdw_corr = self.__VtW_layer[w_next:w_next + w_shape] / (1 - self.__beta_2 ** self.iteration)
                    w_update = self.__learning_rate * (mdw_corr / (np.sqrt(vdw_corr) + self.__epsilon))
                    # AdamRMSP Bias
                    self.__MtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_1 * self.__MtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_1) * dE_dZ
                    self.__VtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_2 * self.__VtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_2) * dE_dZ ** 2
                    vdb_corr = self.__VtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_2 ** self.iteration)
                    mdb_corr = self.__MtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_1 ** self.iteration)
                    b_update = self.__learning_rate * (mdb_corr / (np.sqrt(vdb_corr) + self.__epsilon))
                    # weight update
                    self.__W_layer[w_next:w_next + w_shape] = \
                        self.__W_layer[w_next:w_next + w_shape] - w_update
                    # bias update
                    self.__B_layer[a_next: a_next + a_shape] = \
                        self.__B_layer[a_next: a_next + a_shape] - b_update
                    if self.__RNN and i < len(self.__Layer_shape) - 2:
                        # previous RNN bias reference as it
                        self.__B_layer[a_next + a_shape: a_next + a_shape + a_shape] = self.__B_layer[a_next: a_next + a_shape]
            else:
                if not self.__RNN or (self.__RNN and i % 2 == 1):  # if the past RNN
                    w_update = self.__learning_rate * dE_dW
                    b_update = self.__learning_rate * dE_dZ
                    # weight update
                    self.__W_layer[w_next:w_next + w_shape] = \
                        self.__W_layer[w_next:w_next + w_shape] - w_update
                    # bias update
                    self.__B_layer[a_next: a_next + a_shape] = \
                        self.__B_layer[a_next: a_next + a_shape] - b_update
            # dE_dA PART
            if self.__RNN and i % 2 == 0:
                self.__gE_layer[a_next - self.__Layer_shape[i - 1]: a_next] = dE_dZ
            else:
                self.__gE_layer[a_next - self.__Layer_shape[i - 1]: a_next] = \
                    np.matmul(
                        self.__W_layer[w_next:w_next + w_shape].reshape(self.__Layer_shape[i - 1],
                                                                        self.__Layer_shape[i]),
                        np.transpose(dE_dZ)).flatten()  # transpose multiplication
            # next iteration
            a_next -= self.__Layer_shape[i - 1]
            w_next -= self.__Layer_shape[i - 1] * self.__Layer_shape[i - 2]

    def csv_save(self, file_name) -> None:
        """
        It exports the __W_layer and __B_layer for current neural network
            'file_name_W_layer.csv' 'file_name_B_layer.csv'
        The file will be saved in the same document

        Args:
            file_name(str)

        """
        np.savetxt(file_name + '_W.csv', self.__W_layer, delimiter=',')
        np.savetxt(file_name + '_B.csv', self.__B_layer, delimiter=',')

    def csv_load(self, file_name):
        """
        It imports the __W_layer and __B_layer in current neural network
            'file_name_W_layer.csv' and 'file_name_B_layer.csv'
        The file will be loaded from the same document

        Args:
            file_name(str)

        """
        assert os.path.isfile(file_name + '_W.csv') and os.path.isfile(file_name + '_B.csv')
        self.__W_layer = np.loadtxt(file_name + '_W.csv', delimiter=',')
        self.__B_layer = np.loadtxt(file_name + '_B.csv', delimiter=',')

    def layer_copy(self, __W_layer, __B_layer) -> None:
        """
        Args:
            __W_layer (list)
            __B_layer (list)
        """
        assert len(self.__W_layer) == len(__W_layer) and len(self.__B_layer) == len(__B_layer)
        self.__W_layer = np.copy(__W_layer)
        self.__B_layer = np.copy(__B_layer)

    def run(self, input_val, dropout_rate=0.0) -> None:
        """
        Args:
            input_val(np.ndarray)
            dropout_rate(float):drop out rate
        """
        assert len(input_val) == self.__Layer_shape[0]
        self.__Z_layer[0:self.__Layer_shape[0]] = np.array(input_val)  # input
        self.__drop_Out_rate = dropout_rate
        self.__cpu_run()

    def opt_reset(self) -> None:
        """
        Reset the optimization Layer values
        """
        self.__VtW_layer = np.zeros(self.__VtW_layer.size)
        self.__MtW_layer = np.zeros(self.__MtW_layer.size)
        self.__VtB_layer = np.zeros(self.__VtB_layer.size)
        self.__MtB_layer = np.zeros(self.__MtB_layer.size)
        self.iteration = 1

    def learning_set(self,
                     learning_rate=0.001,
                     dropout_rate=0.0,
                     loss_fun=MSE,
                     learn_optima='ADAMRMSP',
                     processor='NONE') -> None:
        """
        Args:
            learning_rate(float): learning rate
            dropout_rate(float): droup out rate
            loss_fun(function): Cost function types
            learn_optima(str): ADAMRMSP
            processor(str): Process type
        """
        self.__drop_Out_rate = dropout_rate
        self.__learning_rate = learning_rate
        self.__learn_optima = learn_optima
        self.__processor = processor
        self.__loss_fun = loss_fun

    def learn_start(self, out_val, target_val) -> bool:
        """
        * learning_set is required.

        The Nan, inf error don't affect to the weight update

        Args:
            out_val(np.ndarray): the result of network
            target_val(np.ndarray): target value

        Returns:
            True is success for back propagation, False is Nan or Inf detected in update
        """
        assert len(out_val) == self.__Layer_shape[-1]
        assert len(target_val) == self.__Layer_shape[-1]
        # set value
        self.output = out_val
        self.target_val = target_val
        # learn start
        copy_W_layer = np.copy(self.__W_layer)
        # initializing
        self.__gE_layer[-self.__Layer_shape[-1]:], self.error = self.__loss_fun(out_val, target_val)
        self.__cpu_back()
        if np.any(np.isnan(self.__W_layer)) \
                or np.any(np.isinf(self.__W_layer)) \
                or np.any(np.isnan(self.__A_layer)) \
                or np.any(np.isinf(self.__A_layer)) \
                or np.any(np.isnan(self.__VtW_layer)) \
                or np.any(np.isinf(self.__VtW_layer)):
            self.__W_layer = copy_W_layer
            return False
        return True

    def generate_weight(self) -> None:
        """This is generating weight layer by current layer shape.
        """
        self.__W_layer = np.empty(0)
        self.__VtW_layer = np.empty(0)
        self.__MtW_layer = np.empty(0)
        for i in range(len(self.__Layer_shape) - 1):
            W = np.ones(self.__Layer_shape[i] * self.__Layer_shape[i + 1])
            VW = np.zeros(self.__Layer_shape[i] * self.__Layer_shape[i + 1])
            self.__W_layer = np.append(self.__W_layer, W)
            self.__VtW_layer = np.append(self.__VtW_layer, VW)
            self.__MtW_layer = np.append(self.__MtW_layer, VW)

    def xavier_initialization(self) -> None:
        """
        It initializes the weight layers' value by Xavier uniform initialization
        """
        n_in = self.__Layer_shape[0]
        n_out = self.__Layer_shape[-1]
        copy_step = float(n_in + n_out)
        self.__W_layer = np.random.uniform(-sqrt(6 / copy_step), sqrt(6 / copy_step), self.__W_layer.shape[0])

    def he_initialization(self) -> None:
        """
        It initializes the weight layers' value by He uniform initialization
        """
        n_in = self.__Layer_shape[0]
        self.__W_layer = np.random.uniform(-sqrt(6 / n_in), sqrt(6 / n_in), self.__W_layer.shape[0])

    def generate_rnn_set(self):
        """
        The hidden layer should be the RNN structure

        EXAMPLE:
        _______
            Layer shape is [350, 40, 40, 70, 70, 12]
        """
        assert (len(self.__Layer_shape) - 2) % 2 == 0
        a_next = 0
        w_next = 0
        self.__RNN = True
        for i in range(0, len(self.__Layer_shape) - 1):
            a_shape = self.__Layer_shape[i]
            w_shape = self.__Layer_shape[i] * self.__Layer_shape[i + 1]
            if i % 2 == 1:
                assert self.__Layer_shape[i] == self.__Layer_shape[i+1]
                self.__EQ_layer[i] = linear_x
                self.__W_layer[w_next:w_next + w_shape] = np.eye(self.__Layer_shape[i],
                                                                 self.__Layer_shape[i + 1]).flatten()
                self.__B_layer[a_next:a_next + a_shape] = self.__B_layer[a_next + a_shape:a_next + a_shape + self.__Layer_shape[i + 1]]
            a_next += a_shape
            w_next += w_shape

    def show_layer(self):
        w_next = 0
        print(Fore.LIGHTMAGENTA_EX, self.__Layer_shape)
        for i in range(len(self.__Layer_shape) - 1):
            w_shape = self.__Layer_shape[i] * self.__Layer_shape[i + 1]
            print(Fore.LIGHTRED_EX, f'--------W {i}---------\n[{self.__Layer_shape[i]}x{self.__Layer_shape[i + 1]}]')
            print(Fore.LIGHTBLUE_EX,
                  self.__W_layer[w_next:w_next + w_shape].reshape(self.__Layer_shape[i], self.__Layer_shape[i + 1]),
                  Fore.RESET)
            w_next += w_shape

        a_next = 0
        for i in range(len(self.__Layer_shape)):
            print(Fore.LIGHTRED_EX, f'--------A {i}---------\n[{self.__Layer_shape[i]}]')
            a_shape = self.__Layer_shape[i]
            print(Fore.LIGHTYELLOW_EX, self.__A_layer[a_next:a_next + a_shape], Fore.RESET)
            a_next += a_shape

        a_next = 0
        for i in range(len(self.__Layer_shape)):
            print(Fore.LIGHTRED_EX, f'--------B {i}---------\n[{self.__Layer_shape[i]}]')
            a_shape = self.__Layer_shape[i]
            print(Fore.LIGHTCYAN_EX, self.__B_layer[a_next:a_next + a_shape], Fore.RESET)
            a_next += a_shape

    def add_layer(self, number, active_fn=ReLU, normal=linear_x):
        """
        This function add layer in neural network.

        * After :  generate_weight() function is required.

        Args:
            number(int):The # of neuron in layer
            active_fn(function):The activation function
            normal(function):The batch normalization function

        """
        self.__B_layer = np.append(self.__B_layer, np.random.uniform(0, 0.00001, number))
        self.__Z_layer = np.append(self.__Z_layer, np.zeros(number))
        self.__N_layer = np.append(self.__N_layer, normal)
        self.__X_layer = np.append(self.__X_layer, np.zeros(number))
        self.__A_layer = np.append(self.__A_layer, np.zeros(number))
        self.__VtB_layer = np.append(self.__VtB_layer, np.zeros(number))
        self.__MtB_layer = np.append(self.__MtB_layer, np.zeros(number))
        self.__gE_layer = np.append(self.__gE_layer, np.zeros(number))
        self.__EQ_layer = np.append(self.__EQ_layer, active_fn)
        self.__Layer_shape = np.append(self.__Layer_shape, number)
        self.output = np.empty(number)
        self.target_val = np.empty(number)

    def get_shape(self) -> np.ndarray:
        """
        :return: Layer shape array
        """
        return self.__Layer_shape

    def get_layer(self) -> np.ndarray:
        """
        :return: [W_layer, B_layer]
        """
        return np.array([self.__W_layer, self.__B_layer], dtype=object)

    def set_w_layer(self, layer):
        """
        Args:
            layer(np.ndarray[np.double])
        """
        assert len(self.__W_layer) == len(layer)
        self.__W_layer = np.copy(layer)

    def set_b_layer(self, layer):
        """
        Args:
            layer(np.ndarray[np.double])
        """
        assert len(self.__B_layer) == len(layer)
        self.__B_layer = np.copy(layer)
