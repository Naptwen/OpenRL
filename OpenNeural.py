import numpy as np
import sympy as sym
from numpy import ndarray
from sympy import sqrt, diff
from colorama import Fore
from opencl_algorithm import *
import os
from copy import deepcopy


def shannon_entropy(values):
    """
    Args:
        values(np.ndarray): total 1 probability
    :return: Entropy H(x)
    """
    return -np.sum(values * np.log1p(values))


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


def linear_x(values=np.ndarray, gradient=False):
    if not gradient:
        return values
    return np.ones(len(values))


def logp1_x(values=np.ndarray, gradient=False):
    if not gradient:
        return np.log1p(values)
    return np.reciprocal(values + 1)


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


def parametricReLU(values, gradient=False, a=0.01):
    if gradient:
        return np.where(values > 0, 1, a)
    return np.where(values > a * values, values, a * values)


def leakReLU(values, gradient=False):
    if gradient:
        return np.where(values > 0, 1, 0.03)
    return np.where(values > 0.03 * values, values, 0.03 * values)


# NORMALIZATION FUNCTIONS

def znormal(values, gradient=False):
    if not gradient:
        if np.std(values) == 0:
            return np.ones(len(values))
        return (values - np.mean(values)) / (np.std(values))
    return (np.std(values)) * values + np.mean(values)


def min_max_normal(values, gradient=False):
    m, n = max(values), min(values)
    if not gradient:
        if m == n:
            return np.ones(len(values))
        return (values - n) / (m - n)
    if m == n:
        return np.ones(len(values))
    return np.full(len(values), 1 / (m - n))


def bitsoftmax(values, gradient=False):
    f = np.exp(values - np.max(values))
    p = f / np.sum(f)
    if gradient:
        p[np.argmax(f)] -= 1
        return p * values
    return p


def softmax(values, gradient=False):
    f = np.exp(values - max(values))
    p = f / np.sum(f)
    if np.any(np.isnan(p)):
        print(values, f, p)
        exit()
    if gradient:
        return f * (1 - f)
    return p


# def logsoftmax(values, gradient=False):
#     f = np.exp(values)
#     p = np.log(f / np.sum(f))
#     if gradient:
#         return p * (1-p)
#     return p


def drop_Out(layer, drop_per):
    assert 0 <= drop_per < 1
    mask = np.random.uniform(0, 1, layer.shape[0]) > drop_per
    return mask * layer.shape[0] / (1.0 - drop_per)


# LOSS FUNCTIONS

def SIMPLE(x, y):
    return x, y - x


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


def MSE2(x, y):
    """
    Mean Square Error with power 2
    """
    return (x - y), np.sum((y - x) ** 2) / 2


def RMSE(x, y):
    """
    Root Mean Squared Error
    """
    return -(y - x), np.sqrt(np.mean((y - x) ** 2))


def HUBER(x, y):
    h = 1
    return        np.where(np.abs(y - x) <= 1, x - y, h * np.sign(x - y)), \
           np.sum(np.where(np.abs(y - x) <= 1, 0.5 * (y - x) ** 2, h * np.abs(y - x) - 0.5 * h**2))


def PSEUDO_HUBER(x, y):
    a = np.abs(y-x) + 0.00000001
    f = a * sqrt(1 + x**2/a**2)
    return x/f, np.sum(f)

# https://arxiv.org/pdf/2108.12627.pdf
def STRIC_HUBER(x, y):
    pass


def CROSS_ENTROPY(x=np.ndarray, y=np.ndarray):
    """
    Args:
        x(np.ndarray): probability out value
        y(np.ndarray): percentage right(target) value
    """
    assert (y <= 1).all()
    assert (y >= 0).all()
    assert (x <= 1).all()
    assert (x >= 0).all()
    g = (y - x)
    return g, np.sum(y * np.log1p(x))


def JSD(x=np.ndarray, y=np.ndarray):
    """
    Args:
        x(np.ndarray): probability out value
        y(np.ndarray): percentage right(target) value
    """
    assert (y <= 1).all()
    assert (y >= 0).all()
    assert (x <= 1).all()
    assert (x >= 0).all()
    return shannon_entropy(0.5 * (x + y)) - 0.5 * (shannon_entropy(y) + shannon_entropy(x))


def KLD(x=np.ndarray, y=np.ndarray):
    """
    Args:
        x(np.ndarray): probability out value
        y(np.ndarray): percentage right(target) value
    """
    assert (y <= 1).all()
    assert (y >= 0).all()
    assert (x <= 1).all()
    assert (x >= 0).all()
    return -np.sum(y * np.log1p(x / (y + 0.000000001)))


def BINARY_CROSS(x, y):
    """
    Binary Cross Entropy
    """
    return -(  # y * log(a)
            y * np.log1p(x)
            # +(1-y)
            + (np.ones(len(x)) - y)
            # *log(1-a)
            * np.log1p(np.ones(len(x)) - x)
    ), -sum(y * np.log1p(x))


def RELATIVE_ENTROPY(x, y):
    """
    Relative Entropy Error
    """
    return -y / x, np.sum(y * np.log1p(y / x))


class openNeural:
    """
    This is the neural network class by Useop Gim
    After declare class below functions must be

    * add_layer() : adding layer

    * generate_weight() : generating the weight after adding layer

    """
    __W_layer = np.empty(0, dtype=np.float)  # Weight Layer
    __B_layer = np.empty(0, dtype=np.float)  # Bias Layer
    __Z_layer = np.empty(0, dtype=np.float)  # Sum of weight times value Layer
    __X_layer = np.empty(0, dtype=np.float)  # Z + B Layer
    __N_layer = np.empty(0, dtype=object)  # Batch normal Layer
    __A_layer = np.empty(0, dtype=np.float)  # Activation Layer
    __EQ_layer = np.empty(0, dtype=object)  # Equation layer function pointer
    __Layer_shape = np.empty(0, dtype=np.int64)  # it contains Layer shapes information
    __VtW_layer = np.empty(0, dtype=np.float)  # Velocity layer by W for RMSP
    __MtW_layer = np.empty(0, dtype=np.float)  # Momentum layer by W for Adam
    __VtB_layer = np.empty(0, dtype=np.float)  # as same as above but by B
    __MtB_layer = np.empty(0, dtype=np.float)  # as same as above but by B
    __W_UPDATE_layer = np.empty(0, dtype=np.float)  # Weight update Layer
    __B_UPDATE_layer = np.empty(0, dtype=np.float)  # Bias update Layer
    __gE_layer = np.empty(0, dtype=np.float)  # It contains the error for each result and A layer
    __processor = str
    __gradient_clipping_norm = float
    __drop_Out_rate = 0  # drop out rate
    __learning_rate = float
    __learn_optima = 'ADAM'
    __loss_fun = object
    __iteration = 0  # For using the AdamRMSP iteration value
    __beta_1 = float  # For using velocity rate
    __beta_2 = float  # For using momentum rate
    __epsilon = float  # For using velocity rate (to prevent dividing by 0)
    output = np.empty(0)
    target_val = np.empty(0, dtype=np.float)  # It is target value, same size of last value of the layer shape
    error = 1000  # loss function's error

    def __init__(self):
        self.__beta_1 = 0.9
        self.__beta_2 = 0.999
        self.__epsilon = 0.00000001

    def __lshift__(self, trg):
        """
        Args:
            trg(openNeural):
        """
        self.__W_layer =                trg.__W_layer.copy()
        self.__W_UPDATE_layer =         trg.__W_UPDATE_layer.copy()
        self.__B_UPDATE_layer =         trg.__B_UPDATE_layer.copy()
        self.__B_layer =                trg.__B_layer.copy()
        self.__Z_layer =                trg.__Z_layer.copy()
        self.__X_layer =                trg.__X_layer.copy()
        self.__N_layer =                trg.__N_layer.copy()
        self.__A_layer =                trg.__A_layer.copy()
        self.__EQ_layer =               trg.__EQ_layer.copy()
        self.__Layer_shape =            trg.__Layer_shape.copy()
        self.__VtW_layer =              trg.__VtW_layer.copy()
        self.__MtW_layer =              trg.__MtW_layer.copy()
        self.__VtB_layer =              trg.__VtB_layer.copy()
        self.__MtB_layer =              trg.__MtB_layer.copy()
        self.__gE_layer =               trg.__gE_layer.copy()
        self.__processor =              trg.__processor
        self.__gradient_clipping_norm = trg.__gradient_clipping_norm
        self.__drop_Out_rate =          trg.__drop_Out_rate
        self.__learning_rate =          trg.__learning_rate
        self.__learn_optima =           trg.__learn_optima
        self.__loss_fun =               trg.__loss_fun
        self.__iteration =              trg.__iteration
        self.__beta_1 =                 trg.__beta_1
        self.__beta_2 =                 trg.__beta_2
        self.__epsilon =                trg.__epsilon

    def __cpu_run(self) -> np.ndarray:
        """
        This is cpu process forward
        """
        a_next = 0
        w_next = 0
        for i in range(len(self.__Layer_shape)):
            a_shape = self.__Layer_shape[i]
            self.__X_layer[a_next: a_next + a_shape] \
                = self.__Z_layer[a_next: a_next + a_shape] + self.__B_layer[a_next: a_next + a_shape]
            # normalization
            self.__X_layer[a_next: a_next + a_shape] = self.__N_layer[i](self.__X_layer[a_next: a_next + a_shape])
            # activation function part
            self.__A_layer[a_next: a_next + a_shape] = self.__EQ_layer[i](self.__X_layer[a_next: a_next + a_shape])
            # checking drop out
            if self.__drop_Out_rate != 0 and i < len(self.__Layer_shape) - 1:
                self.__A_layer[a_next: a_next + a_shape] = drop_Out(self.__A_layer[a_next: a_next + a_shape],
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
        self.output = self.__A_layer[-self.__Layer_shape[-1]:].copy()
        return self.output.copy() # deepcopy value

    def __cpu_back(self) -> None:
        """
        This is cpu process back propagation
        """
        # learn start
        copy_w_layer = self.__W_layer.copy()
        copy_b_layer = self.__B_layer.copy()
        self.__iteration += 1
        a_next = len(self.__A_layer) - self.__Layer_shape[-1]
        w_next = len(self.__W_layer) - self.__Layer_shape[-1] * self.__Layer_shape[-2]
        for i in reversed(range(1, len(self.__Layer_shape))):
            try:
                a_shape = self.__Layer_shape[i]
                w_shape = self.__Layer_shape[i] * self.__Layer_shape[i - 1]
                dE_dA = self.__gE_layer[a_next: a_next + a_shape]
                # gradient clipping
                if self.__gradient_clipping_norm > 0:
                    dE_dA = self.gradient_clipping_norm(dE_dA, self.__gradient_clipping_norm)
                # calculate partial derivative of A
                dA_dNX = self.__EQ_layer[i](self.__X_layer[a_next: a_next + a_shape], True)
                # batch normal recovery
                dNX_dX = self.__N_layer[i](self.__X_layer[a_next: a_next + a_shape], True)
                # dX_dZ = 1
                dE_dZ = dE_dA * dA_dNX * dNX_dX
                # converting dZ_dA for multiplication
                repeat_dZ_dA = np.repeat([self.__A_layer[a_next - self.__Layer_shape[i - 1]: a_next]], repeats=a_shape,
                                         axis=0)
                dig_dE_dZ = np.diag(dE_dZ)
                # gradient dE_dW
                dE_dW = np.matmul(dig_dE_dZ, repeat_dZ_dA).transpose().flatten()  # (dig * repeat)^T is [a,b][2a 2b][3a 3b]
                if self.__learn_optima == 'ADAM':
                    # AdamRMSP Weight
                    self.__MtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_1 * self.__MtW_layer[w_next:w_next + w_shape] + (
                                    1 - self.__beta_1) * dE_dW  # pm + (1-p)g
                    self.__VtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_2 * self.__VtW_layer[w_next:w_next + w_shape] + (
                                    1 - self.__beta_2) * dE_dW ** 2  # pv + (1-p)g
                    mdw_corr = self.__MtW_layer[w_next:w_next + w_shape] / (
                                1 - self.__beta_1 ** self.__iteration)  # m/(1-p)
                    vdw_corr = self.__VtW_layer[w_next:w_next + w_shape] / (
                                1 - self.__beta_2 ** self.__iteration)  # v/(1-p)
                    w_update = self.__learning_rate * (mdw_corr / (np.sqrt(vdw_corr) + self.__epsilon))
                    # AdamRMSP Bias
                    self.__MtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_1 * self.__MtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_1) * dE_dZ
                    self.__VtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_2 * self.__VtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_2) * dE_dZ ** 2
                    mdb_corr = self.__MtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_1 ** self.__iteration)
                    vdb_corr = self.__VtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_2 ** self.__iteration)
                    b_update = self.__learning_rate * (mdb_corr / (np.sqrt(vdb_corr) + self.__epsilon))
                elif self.__learn_optima == 'NADAM':
                    # AdamRMSP Weight
                    self.__MtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_1 * self.__MtW_layer[w_next:w_next + w_shape] + (1 - self.__beta_1) * dE_dW
                    self.__VtW_layer[w_next:w_next + w_shape] = \
                        self.__beta_2 * self.__VtW_layer[w_next:w_next + w_shape] + (1 - self.__beta_2) * dE_dW ** 2
                    mdw_corr = self.__MtW_layer[w_next:w_next + w_shape] / (1 - self.__beta_1 ** self.__iteration)
                    vdw_corr = self.__VtW_layer[w_next:w_next + w_shape] / (1 - self.__beta_2 ** self.__iteration)
                    # Nesterov
                    mdw_corr = self.__beta_1 * mdw_corr + (1 - self.__beta_1) * dE_dW
                    w_update = self.__learning_rate * (mdw_corr / (np.sqrt(vdw_corr) + self.__epsilon))
                    # AdamRMSP Bias
                    self.__MtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_1 * self.__MtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_1) * dE_dZ
                    self.__VtB_layer[a_next: a_next + a_shape] = \
                        self.__beta_2 * self.__VtB_layer[a_next: a_next + a_shape] + (1 - self.__beta_2) * dE_dZ ** 2
                    mdb_corr = self.__MtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_1 ** self.__iteration)
                    # Nesterov
                    mdb_corr = self.__beta_1 * mdb_corr + (1 - self.__beta_1) * dE_dZ
                    vdb_corr = self.__VtB_layer[a_next: a_next + a_shape] / (1 - self.__beta_2 ** self.__iteration)
                    b_update = self.__learning_rate * (mdb_corr / (np.sqrt(vdb_corr) + self.__epsilon))
                else:
                    w_update = self.__learning_rate * dE_dW
                    b_update = self.__learning_rate * dE_dZ
                self.__W_UPDATE_layer[w_next:w_next + w_shape] = w_update
                self.__B_UPDATE_layer[a_next: a_next + a_shape] = b_update
                # weight update
                self.__W_layer[w_next:w_next + w_shape] = \
                    self.__W_layer[w_next:w_next + w_shape] - w_update
                # bias update
                self.__B_layer[a_next: a_next + a_shape] = \
                    self.__B_layer[a_next: a_next + a_shape] - b_update
                # dE_dA PART
                self.__gE_layer[a_next - self.__Layer_shape[i - 1]: a_next] = \
                    np.matmul(
                        self.__W_layer[w_next:w_next + w_shape].reshape(self.__Layer_shape[i - 1],
                                                                        self.__Layer_shape[i]),
                        np.transpose(dE_dZ)).flatten()  # transpose multiplication
                # next iteration
                a_next -= self.__Layer_shape[i - 1]
                w_next -= self.__Layer_shape[i - 1] * self.__Layer_shape[i - 2]
            except (ZeroDivisionError, OverflowError) as e:
                self.__W_layer = deepcopy(copy_w_layer)
                self.__B_layer = deepcopy(copy_b_layer)
                print(f"Layer {i} has error, learning dismissed: ", e)

    def run(self, input_val, dropout_rate=0.0) -> ndarray:
        """
        Args:
            input_val(np.ndarray)
            dropout_rate(float):drop out rate
        """
        assert len(input_val) == self.__Layer_shape[0]
        assert 0 <= dropout_rate <= 1
        self.__Z_layer[0:self.__Layer_shape[0]] = np.array(input_val)  # input
        self.__drop_Out_rate = dropout_rate
        return self.__cpu_run()

    def opt_reset(self) -> None:
        """
        Reset the optimization Layer values
        """
        self.__VtW_layer = np.zeros(self.__VtW_layer.size)
        self.__MtW_layer = np.zeros(self.__MtW_layer.size)
        self.__VtB_layer = np.zeros(self.__VtB_layer.size)
        self.__MtB_layer = np.zeros(self.__MtB_layer.size)
        self.__iteration = 1

    def gradient_clipping_value(self, grad_list, max_threshold, min_threshold):
        grad_list = np.where(np.linalg.norm(grad_list) > max_threshold, max_threshold)
        grad_list = np.where(np.linalg.norm(grad_list) > min_threshold, min_threshold)
        return grad_list

    def gradient_clipping_norm(self, grad_list, threshold):
        grad_list = np.where(np.linalg.norm(grad_list) > threshold,
                             threshold * grad_list / (np.linalg.norm(grad_list) + 0.00000001), grad_list)
        return grad_list

    def learning_set(self,
                     gradient_clipping_norm=0.0,
                     learning_rate=0.001,
                     dropout_rate=0.0,
                     loss_fun=MSE2,
                     learn_optima='ADAM',
                     processor='NONE') -> None:
        """
        Args:
            gradient_clipping_norm(float): gradient cliiping for norm (0 is Not use)
            learning_rate(float): learning rate
            dropout_rate(float): droup out rate
            loss_fun(function): Cost function types
            learn_optima(str): ADAM, NADAM, None
            processor(str): Process type
        """
        assert 0 <= gradient_clipping_norm
        self.__gradient_clipping_norm = gradient_clipping_norm
        self.__drop_Out_rate = dropout_rate
        self.__learning_rate = learning_rate
        self.__learn_optima = learn_optima
        self.__processor = processor
        self.__loss_fun = loss_fun

    def learn_start(self, out_val, target_val, direct_gradient=None) -> bool:
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

        # initializing
        if direct_gradient is None:
            self.__gE_layer[-self.__Layer_shape[-1]:], self.error = self.__loss_fun(out_val, target_val)
        else:
            self.__gE_layer[-self.__Layer_shape[-1]:] = direct_gradient
        self.__cpu_back()
        return True

    def generate_weight(self) -> None:
        """This is generating weight layer by current layer shape.
        """
        self.__W_layer = np.empty(0)
        self.__VtW_layer = np.empty(0)
        for i in range(len(self.__Layer_shape) - 1):
            W = np.ones(self.__Layer_shape[i] * self.__Layer_shape[i + 1])
            VW = np.zeros(self.__Layer_shape[i] * self.__Layer_shape[i + 1])
            self.__W_layer = np.append(self.__W_layer, W)
            self.__VtW_layer = np.append(self.__VtW_layer, VW)
        self.__MtW_layer = np.copy(self.__VtW_layer)
        self.__W_UPDATE_layer = np.copy(self.__VtW_layer)
        self.__B_UPDATE_layer = np.zeros(len(self.__A_layer))

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

    def get_update_layer(self) -> np.ndarray:
        """
        :return: [W_update_layer, B_ypdate_layer]
        """
        return np.array([self.__W_UPDATE_layer, self.__B_UPDATE_layer], dtype=object)

    def set_w_layer(self, layer):
        """
        Args:
            layer(np.ndarray[np.float])
        """
        assert len(self.__W_layer) == len(layer)
        self.__W_layer = layer.copy()

    def set_b_layer(self, layer):
        """
        Args:
            layer(np.ndarray[np.float])
        """
        assert len(self.__B_layer) == len(layer)
        self.__B_layer = layer.copy()

    def numpy_save(self, file_name) -> None:
        """
        It exports the __W_layer and __B_layer for current neural network
            'file_name_W_layer.csv' 'file_name_B_layer.csv'
        The file will be saved in the same document

        Args:
            file_name(str)

        """
        np.savetxt(file_name + '_W.csv', self.__W_layer)
        np.savetxt(file_name + '_B.csv', self.__B_layer)
        np.savetxt(file_name + '_Z.csv', self.__Z_layer)
        np.savetxt(file_name + '_X.csv', self.__X_layer)
        np.savetxt(file_name + '_A.csv', self.__A_layer)
        np.savetxt(file_name + '_L.csv', self.__Layer_shape, fmt="%d")
        np.savetxt(file_name + '_Vw.csv', self.__VtW_layer)
        np.savetxt(file_name + '_Mw.csv', self.__MtW_layer)
        np.savetxt(file_name + '_Vb.csv', self.__VtB_layer)
        np.savetxt(file_name + '_Mb.csv', self.__MtB_layer)
        np.savetxt(file_name + '_Wu.csv', self.__W_UPDATE_layer)
        np.savetxt(file_name + '_Bu.csv', self.__B_UPDATE_layer)
        np.savetxt(file_name + '_E.csv', self.__gE_layer)
        eq_list = np.empty(0)
        for T in self.__EQ_layer:
            eq_list = np.append(eq_list, T.__name__)
        n_list = np.empty(0)
        for N in self.__N_layer:
            n_list = np.append(n_list, N.__name__)
        np.save(file_name + '_Q', eq_list) # function
        np.save(file_name + '_N', n_list) # function


    def numpy_load(self, file_name):
        """
        It imports the __W_layer and __B_layer in current neural network
            'file_name_W_layer.csv' and 'file_name_B_layer.csv'
        The file will be loaded from the same document

        Args:
            file_name(str)

        """
        self.__W_layer=np.loadtxt(file_name + '_W.csv')
        self.__B_layer=np.loadtxt(file_name + '_B.csv')
        self.__Z_layer=np.loadtxt(file_name + '_Z.csv')
        self.__X_layer=np.loadtxt(file_name + '_X.csv')
        self.__A_layer=np.loadtxt(file_name + '_A.csv')
        self.__Layer_shape=np.loadtxt(file_name + '_L.csv', dtype=int)
        self.__VtW_layer=np.loadtxt(file_name + '_Vw.csv')
        self.__MtW_layer=np.loadtxt(file_name + '_Mw.csv')
        self.__VtB_layer=np.loadtxt(file_name + '_Vb.csv')
        self.__MtB_layer=np.loadtxt(file_name + '_Mb.csv')
        self.__W_UPDATE_layer=np.loadtxt(file_name + '_Wu.csv')
        self.__B_UPDATE_layer=np.loadtxt(file_name + '_Bu.csv')
        self.__gE_layer=np.loadtxt(file_name + '_E.csv')

        eq_list = np.load(file_name + '_Q.npy')
        n_list = np.load(file_name + '_N.npy')
        self.__EQ_layer = np.empty(0)

        for eq_str in eq_list:
            if eq_str == linear_x.__name__ :
                self.__EQ_layer = np.append(self.__EQ_layer, linear_x)
            elif eq_str == leakReLU.__name__ :
                self.__EQ_layer = np.append(self.__EQ_layer, leakReLU)
            elif eq_str == ReLU.__name__ :
                self.__EQ_layer = np.append(self.__EQ_layer, ReLU)
            elif eq_str == logp1_x.__name__ :
                self.__EQ_layer = np.append(self.__EQ_layer, logp1_x)

        self.__N_layer = np.empty(0)
        for n_str in n_list:
            if n_str == linear_x.__name__ :
                self.__N_layer = np.append(self.__N_layer, linear_x)
            elif n_str == znormal.__name__ :
                self.__N_layer = np.append(self.__N_layer, znormal)
            elif n_str == min_max_normal.__name__ :
                self.__N_layer = np.append(self.__N_layer, parametricReLU)
            elif n_str == softmax.__name__ :
                self.__N_layer = np.append(self.__N_layer, softmax)

    def layer_copy(self, __W_layer, __B_layer) -> None:
        """
        Args:
            __W_layer (list)
            __B_layer (list)
        """
        assert len(self.__W_layer) == len(__W_layer) and len(self.__B_layer) == len(__B_layer)
        self.__W_layer = np.copy(__W_layer)
        self.__B_layer = np.copy(__B_layer)
