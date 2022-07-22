import random
import time
import sympy as sym
from sympy import sqrt, diff
import numpy as np
import os

def softmax(values):
    val2exp = np.exp(values)
    return val2exp / np.sum(val2exp)

def HyperBolic(values):
    ex = np.exp(values)
    e_x = np.exp(-values)
    return  (ex - e_x) / (ex + e_x)

def sigmoid(values):
    values = np.exp(-values)
    return 1 / (1 + values)

def normalminmax(values):
    values = np.array(values)
    if np.min(values) != np.max(values):
        val2norm = (values - np.min(values))/(np.max(values) - np.min(values))
        return 2 * val2norm - 1
    return np.zeros(len(values))

def znormal(values):
    normalized = np.empty(0)
    for value in values:
        normalized_num = (value - np.mean(values)) / np.std(values)
        normalized  = np.append(normalized, normalized_num)
    return normalized

def max_min_lmit(values):
    values = np.where(values < -2, -2, values)
    values = np.where(values > 2, 2, values)
    return values

def dropOut(layer, drop_per):
    assert 0 <= drop_per < 1
    mask = np.random.uniform(0, 1, layer.shape[0]) > drop_per
    return mask * layer.shape[0] / (1.0 - drop_per)

class openNeural:
    """
    This is the neural network class\n
    After declare, follow below process \n
    [1]. add_layer : adding layer as much as you want (at least 2 times are required for input layer and output layer)\n
    [2]. generate_weight : After finish the adding layer, must generate weight to create the weight layer\n
    [3]. xavier_initialization : initializing the weight layer by xavier
    """
    W_layer = np.empty(0, dtype = np.double) # Weight Layer
    B_layer = np.empty(0, dtype = np.double) # Basis Layer
    Z_layer = np.empty(0, dtype = np.double)
    X_layer = np.empty(0, dtype = np.double)
    A_layer = np.empty(0, dtype = np.double)
    EQ_layer = np.empty(0, dtype = np.double) # Equation layer, it is only used if interpreting were true
    DE_layer = np.empty(0, dtype = np.double) # Derivative Equation layer, it is only used if interpreting were true
    Layer_shape = np.empty(0, dtype = np.int64) # it contains Layer shapes information
    VtW_layer = np.empty(0, dtype = np.double) # Velocity layer by W for RMSP
    MtW_layer = np.empty(0, dtype = np.double) # Momentum layer by W for Adam
    VtB_layer = np.empty(0, dtype = np.double) # as same as above but by B
    MtB_layer = np.empty(0, dtype = np.double) # as same as above but by B
    gE_layer = np.empty(0, dtype = np.double) # It contains the error for each result and A layer
    Limit_layer = np.empty(0, dtype = np.double) # it is used for limit the output value of A layer
    target_val = np.empty(0, dtype = np.double) # It is target value, the size is as same as the last value of the layer shape
    Error_optimaization = 'NONE'
    drop_out = 0.0 # drop out rate
    error = 1000 # loss function's error
    learning_rate = 0.01
    interpreting = False # it is using for when forward and backward, do the sympy's equation calculating or not
    step = 1 # For using the AdamRMSP iteration value
    beta_1 = 0.9 # For using velocity rate
    beta_2 = 0.9 # For using momentum rate
    epsilon = 0.00000001 # For using velocity rate (to prevent dividing by 0)
    adamRMSP = True
    output = np.empty(0)

    def cpu_run(self):
        """
       It is calculating forward propagation and save each values in Z,X,A layers

       The output should be saved in last layer's A  and also output value

       If interpreting is True then interpreting equation as user input at add layer function

       It must take much longer than given functions
        """
        a_next = 0
        w_next = 0
        for i in range(len(self.Layer_shape)):
            a_shape = self.Layer_shape[i]
            #linearly sum
            self.X_layer[a_next: a_next + a_shape] \
                = self.Z_layer[a_next: a_next + a_shape] + self.B_layer[a_next: a_next + a_shape]
            # activation function part
            if not self.interpreting:
                if self.EQ_layer[i] == 'softmax':  # normalize the output value
                    self.A_layer[a_next: a_next + a_shape] = softmax(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'softmax_normal':  # normalize the output value
                    self.A_layer[a_next: a_next + a_shape] = softmax_normal(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'max_min_limit':  # normalize the values between -1, 1, always the max = 1 and min = -1
                    self.A_layer[a_next: a_next + a_shape] = max_min_limit(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'normal':  # normalize the output value without consider variance
                    self.A_layer[a_next: a_next + a_shape] = normalminmax(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'znormal':  # normalize the output value consider variance within -1, 1
                    self.A_layer[a_next: a_next + a_shape] = znormal(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'sigmoid':  # The typical activation function make between 0 1
                    self.A_layer[a_next: a_next + a_shape] = sigmoid(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'tanh(x)':  # The typical -1, 1 activation function
                    self.A_layer[a_next: a_next + a_shape] = HyperBolic(
                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'ReLU':  # Fast and prevent gradient vanishing
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0, self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'leakReLU':  # prevent dead neuron
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0.3 * self.X_layer[a_next: a_next + a_shape],
                                                                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'parametricReLU':  # prevent dead neuron and also negative flliping
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(-0.3 * self.X_layer[a_next: a_next + a_shape],
                                                                        self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'ReLU:1.0':  # Fast and prevent gradient vanishing
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0, self.X_layer[a_next: a_next + a_shape])
                    self.A_layer[a_next: a_next + a_shape] = np.where(self.A_layer[a_next: a_next + a_shape] > 1, 1, self.A_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'leakReLU:1.0':  # prevent dead neuron
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0.3 * self.X_layer[a_next: a_next + a_shape],
                                                                        self.X_layer[a_next: a_next + a_shape])
                    self.A_layer[a_next: a_next + a_shape] = np.where(self.A_layer[a_next: a_next + a_shape] > 1, 1, self.A_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'parametricReLU:1.0':  # prevent dead neuron and also negative flliping
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(-0.3 * self.X_layer[a_next: a_next + a_shape],
                                                                        self.X_layer[a_next: a_next + a_shape])
                    self.A_layer[a_next: a_next + a_shape] = np.where(self.A_layer[a_next: a_next + a_shape] > 1, 1, self.A_layer[a_next: a_next + a_shape])
                else:
                    self.A_layer[a_next: a_next + a_shape] = self.X_layer[a_next: a_next + a_shape]
            else:
                v_x = sym.Symbol('x')
                self.A_layer[a_next: a_next + a_shape] = np.array(
                    [float(sym.simplify(eq).subs(v_x, val)) for eq, val in
                     zip(self.EQ_layer[i],self.Z_layer[a_next: a_next + a_shape])])
            # checking drop out
            if self.drop_out != 0 and i < len(self.Layer_shape) - 2:
                self.A_layer[a_next: a_next + a_shape] = dropOut(self.A_layer[a_next: a_next + a_shape], self.drop_out)
            elif self.drop_out == 1 and i < len(self.Layer_shape) - 2:
                self.A_layer[a_next: a_next + a_shape] = np.zeors(self.A_layer[a_next: a_next + a_shape].shape[0])
            #obtained multiply weight
            if i < len(self.Layer_shape) - 1:
                w_shape = self.Layer_shape[i] * self.Layer_shape[i + 1]
                self.Z_layer[a_next + a_shape: a_next + a_shape + self.Layer_shape[i+1]] =\
                    np.matmul( self.A_layer[a_next: a_next + a_shape],
                               self.W_layer[w_next: w_next + w_shape].reshape(self.Layer_shape[i],  self.Layer_shape[i + 1]) ).flatten()
                w_next += w_shape
            a_next += a_shape
            self.output = self.A_layer[-self.Layer_shape[-1]:]

    def cpu_back_propagation(self):
        """
        Back propagation, if the interpreting is False then the dA_dZ is based on the derivative of typed functions (default is x)

        If True, it follows the derivative function by sympy (it makes the delay of calculating)
         """
        a_next = len(self.A_layer) - self.Layer_shape[-1]
        w_next = len(self.W_layer) - self.Layer_shape[-1] * self.Layer_shape[-2]
        for i in reversed(range(1, len(self.Layer_shape))):
            a_shape = self.Layer_shape[i]
            w_shape = self.Layer_shape[i] * self.Layer_shape[i - 1]
            dE_dA = self.gE_layer[a_next: a_next + a_shape]
            # calculate partial derivative of A
            dA_dX = np.ones(a_shape)  # typical x
            if self.interpreting:
                dA_dX = np.empty(0)
                for k in range(a_shape):
                    v_x = sym.Symbol('x')
                    val = self.Z_layer[a_next + k]
                    dA_dX = np.append(dA_dX, float(self.DE_layer[i].subs(v_x, val).evalf()))
            else:
                if self.EQ_layer[i] == 'softmax':
                    dA_dX = self.A_layer[a_next: a_next + a_shape] - self.A_layer[a_next: a_next + a_shape]**2
                elif self.EQ_layer[i] == 'sigmoid':
                    dA_dX = self.X_layer[a_next: a_next + a_shape] * (1 - self.X_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == 'tanh(x)':
                    dA_dX = HyperBolic(self.X_layer[a_next: a_next + a_shape]) ** 2
                elif self.EQ_layer[i] == 'ReLU':
                    dA_dX = np.where(self.X_layer[a_next: a_next + a_shape]> 0, 1, 0)
                elif self.EQ_layer[i] == 'parametricReLU':
                    dA_dX = np.where(self.X_layer[a_next: a_next + a_shape] > 0, 1, -0.3)
                elif self.EQ_layer[i] == 'leakReLU':
                    dA_dX = np.where(self.X_layer[a_next: a_next + a_shape] > 0, 1, 0.3)
            dE_dZ = dE_dA * dA_dX
            #converting dZ_dA for multiplication
            repeat_dZ_dA = np.repeat([self.A_layer[a_next - self.Layer_shape[i - 1] : a_next]], repeats = a_shape, axis = 0)
            dig_dE_dZ = np.diag(dE_dZ)
            #gradient dE_dW
            dE_dW = np.matmul(dig_dE_dZ, repeat_dZ_dA).transpose().flatten() #(dig * repeat)^T is [a,b][2a 2b][3a 3b]
            if self.adamRMSP:
                # AdamRMSP Weight
                self.MtW_layer[w_next:w_next + w_shape] =\
                    self.beta_1 * self.MtW_layer[w_next:w_next + w_shape] + (1 - self.beta_1) * dE_dW
                self.VtW_layer[w_next:w_next + w_shape] =\
                    self.beta_2 * self.VtW_layer[w_next:w_next + w_shape] + (1 - self.beta_2) * dE_dW**2
                mdw_corr = self.MtW_layer[w_next:w_next + w_shape] / (1 - self.beta_1 ** self.step)
                vdw_corr = self.VtW_layer[w_next:w_next + w_shape] / (1 - self.beta_2 ** self.step)
                w_update = self.learning_rate * (mdw_corr/(np.sqrt(vdw_corr) + self.epsilon))
                # AdamRMSP Bias
                self.MtB_layer[a_next: a_next + a_shape] = \
                    self.beta_1 * self.MtB_layer[a_next: a_next + a_shape] + (1 - self.beta_1) * dE_dZ
                self.VtB_layer[a_next : a_next + a_shape] =\
                    self.beta_2 * self.VtB_layer[a_next : a_next + a_shape] + (1 - self.beta_2) * dE_dZ**2
                vdb_corr = self.VtB_layer[a_next : a_next + a_shape] / (1 - self.beta_2 ** self.step)
                mdb_corr = self.MtB_layer[a_next : a_next + a_shape] / (1 - self.beta_1 ** self.step)
                b_update = self.learning_rate * (mdb_corr/(np.sqrt(vdb_corr) + self.epsilon))
            else:
                w_update = self.learning_rate * dE_dW
                b_update = self.learning_rate * dE_dZ
            # weight update
            self.W_layer[w_next:w_next + w_shape] =\
                self.W_layer[w_next:w_next + w_shape] - w_update
            # bias update
            self.B_layer[a_next : a_next + a_shape] =\
                self.B_layer[a_next : a_next + a_shape] - b_update
            # dE_dA
            self.gE_layer[a_next - self.Layer_shape[i - 1]: a_next] =\
                np.matmul(self.W_layer[w_next:w_next + w_shape].reshape(self.Layer_shape[i-1],self.Layer_shape[i]),
                          np.transpose(dE_dZ) ).flatten() #transpose multiplication
            #next iteration
            a_next -= self.Layer_shape[i - 1]
            w_next -= self.Layer_shape[i - 1] * self.Layer_shape[i - 2]
        self.step += 1

    # def cl_run(self):
    #     """
    #     It is calculating forward propagation and save each values in Z,A layers\n
    #     The output should be saved in last layer's A layer\n
    #     If interpreting is True then interpreting equation as user input at add layer function\n
    #     It must take much longer than default function leakReLU\n
    #     """
    #     pass # latter too lazy to do it again
    #
    # def cl_back_propagation(self):
    #     """
    #     Back propagation, if the interpreting is False then the dA_dZ is based on the derivative of soft ReLU (default)\n
    #     If not, it follows the derivative function by sympy (it makes the delay of calculating)
    #     """
    #     pass # too lazy to do it again

    def csv_save(self, file_name):
        """
        It only exports the W_layer and B_layer for current neural network

        :param file_name: it is file name for export the file should be written as "file_name_W.csv" and "file_name_B.csv"
        """
        np.savetxt(file_name + '_W.csv', self.W_layer, delimiter = ',')
        np.savetxt(file_name + '_B.csv', self.B_layer, delimiter=',')

    def csv_load(self, file_name):
        """
        It is overlapping the W and B layer of current neural network

        The overlapped neural network's W and B layer should be generated and also have the same shape

        :param file_name: it is file name which is in the same directory with this python script
        If file is not exists, it will not load the file
        """
        if os.path.isfile(file_name + '_W.csv') and os.path.isfile(file_name + '_B.csv'):
            self.W_layer = np.loadtxt(file_name + '_W.csv', delimiter = ',')
            self.B_layer = np.loadtxt(file_name + '_B.csv', delimiter=',')
        else:
            print(Fore.LIGHTRED_EX + ' FILE IS NOT EXITS' + Fore.RESET)

    def run_init(self, input_val, dropout_rate = 0.0):
        """
        :param input_val: it is the input value which will be contained in the first Z_layer
        :param dropout: it is the drop out rate between [0~1], recommend 0 when doing reinforcement learning and for just checking
        """
        self.Z_layer[0:self.Layer_shape[0]] = np.array(input_val)[0:self.Layer_shape[0]] #input
        self.drop_out = dropout_rate

    def learning_reset(self):
        self.VtW_layer = np.zeros(self.VtW_layer.size)
        self.MtW_layer = np.zeros(self.MtW_layer.size)
        self.VtB_layer = np.zeros(self.VtB_layer.size)
        self.MtB_layer = np.zeros(self.MtB_layer.size)
        self.step = 1

    def learning_set(self, learning_rate  = 0.001, dropout_rate = 0.0, loss_fun = 'RMSE', adam_rmsp = True, Error_optimaization = 'NONE'):
        """
        :param learning_rate: learning rate
        :param dropout_rate: droup out rate
        :param loss_fun: Cost function types; DIRECT is just directly put the error, it will not do the forward for checking error
        :param adam_rmsp: this is optimization for the back propagation
        :param Error_optimaization: this is optimization for the Error
        """
        self.drop_out = dropout_rate
        self.learning_rate = learning_rate
        self.adamRMSP = adam_rmsp
        self.Error_optimaization = Error_optimaization
        self.loss_fun = loss_fun

    def learn_start(self, input_val, target_val, show_result = False):
        """
        1 learning_reset is required. if tou were not change the step(iteration), only once.

        2 learning_set is required. if you were not change dropout rate, learning rate during the program, only once.

        3 learning_start can be as much as you want (one learn_start do one cycle forward->Error->backpropagation).

        The Nan, inf error don't affect to the weight update, it will be just ignored but taking process to measure it.

        :param input_val: innput value should be as same as the number of input layer nodes
        :param target_val: target value should be as same as the number of output layer nodes
        :param show_result: If you want to see the infomaration of error and output value after the back propgation
        """
        # initializing
        self.Z_layer[0:self.Layer_shape[0]] = np.array(input_val)[0:self.Layer_shape[0]] #input
        self.target_val = np.array(target_val)[0:self.Layer_shape[-1]] #target
        # learn start
        copy_W_layer = np.copy(self.W_layer)
        start = time.time()
        self.step += 1 # it is used for the adamRMSP, if you need reset for adamRMSP, reset this before learning as 1
        if self.loss_fun != 'DIRECT':
            self.gE_layer[-self.Layer_shape[-1]:] = self.target_val[0:self.Layer_shape[-1]]
        else:
            self.cpu_run()
            if self.loss_fun == 'RMSE':
                self.RMSE()
            elif self.loss_fun == 'MPE':
                self.MPE()
            elif self.loss_fun == 'CROSS':
                self.CROSS_ENTROPY()
            elif self.loss_fun == 'BINARY_CROSS':
                self.BINARY_CROSS()
            elif self.loss_fun == 'RELATIVE_ENTROPY':
                self.RELATIVE_ENTROPY()
            elif self.loss_fun == 'DIRECT':
                self.RELATIVE_ENTROPY()
            else:
                self.MSE()
        self.cpu_back_propagation()
        end = time.time()
        if show_result:
            print(self.step, self.error, self.A_layer[-self.Layer_shape[-1]:])
        if np.any(np.isnan(self.W_layer)) \
                or np.any(np.isinf(self.W_layer)) \
                or np.any(np.isnan(self.A_layer)) \
                or np.any(np.isinf(self.A_layer)) \
                or np.any(np.isnan(self.VtW_layer))\
                or np.any(np.isinf(self.VtW_layer)):
            self.W_layer = copy_W_layer
            if show_result:
                self.learn_show('Cyan', end - start)

    def get_error(self, input_val, target_val):
        """
        :param input_val:
        :param target_val:
        :return: error, if loss_fun is set as DIRECT, it will return jsut the target_val
        """
        self.Z_layer[0:self.Layer_shape[0]] = np.array(input_val)[0:self.Layer_shape[0]]  # input
        self.target_val = np.array(target_val)[0:self.Layer_shape[-1]] #target
        if self.loss_fun != 'DIRECT':
            return self.target_val
        else:
            self.cpu_run()
            if self.loss_fun == 'RMSE':
                self.RMSE()
            elif self.loss_fun == 'MPE':
                self.MPE()
            elif self.loss_fun == 'CROSS':
                self.CROSS_ENTROPY()
            elif self.loss_fun == 'BINARY_CROSS':
                self.BINARY_CROSS()
            elif self.loss_fun == 'RELATIVE_ENTROPY':
                self.RELATIVE_ENTROPY()
            else:
                self.MSE()
        return self.error

    def learn_show(self, color, time):
        np.set_printoptions(precision = 4)
        if color == 'Green':
            print(Fore.GREEN + f'[save] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.step} '
                    f'trg :{self.target_val}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val - self.A_layer[-self.Layer_shape[-1]:], 2)} '
                    f'err :{self.error:.4f}% '
                    f'tim :{time :.2f}s'
                    + Fore.RESET)
        elif color == 'Red':
            print(Fore.RED + f'[save] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.step} '
                    f'trg :{self.target_val}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val - self.A_layer[-self.Layer_shape[-1]:], 2)} '
                    f'err :{self.error:.4f}% '
                    f'tim :{time :.2f}s'
                    + Fore.RESET)
        elif color == 'Cyan':
            print(Fore.CYAN + f'[error] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.step} '
                    f'trg :{self.target_val}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val - self.A_layer[-self.Layer_shape[-1]:], 2)} '
                    f'err :{self.error:.4f}% '
                    f'tim :{time :.2f}s'
                    + Fore.RESET)

    def xavier_initialization(self):
        """
        It initializes the weight layers' value by Xavier initialization
        """
        n_in = self.Layer_shape[0]
        n_out = self.Layer_shape[-1]
        N = float(n_in + n_out)
        self.W_layer = np.random.uniform(-sqrt(6 / N), sqrt(6 / N), self.W_layer.shape[0])

    def generate_weight(self):
        """
        Generate Weight layers by the shape of current Layer\n
        It is required after completing to construct the layer shapes\n
        """
        self.W_layer = np.empty(0)
        self.VtW_layer = np.empty(0)
        self.MtW_layer = np.empty(0)
        for i in range(len(self.Layer_shape) - 1):
            W = np.ones( self.Layer_shape[i] * self.Layer_shape[i + 1] )
            VW = np.zeros( self.Layer_shape[i] * self.Layer_shape[i + 1] )
            self.W_layer = np.append(self.W_layer, W)
            self.VtW_layer = np.append(self.VtW_layer, VW)
            self.MtW_layer = np.append(self.MtW_layer, VW)

    def add_layer(self, number, active_fn):
        """
        Add layer\n
        :param number: The # of neuron in layer
        :param active_fn: 'softmax', 'softmax_normal', 'max_min_limit', 'normal', 'znormal',
        'sigmoid', 'tanh(x)', 'ReLU', 'leakReLU', 'parametricReLU' if not set then it works aas 'x'
        """
        self.Z_layer = np.append(self.Z_layer, np.zeros(number))
        self.X_layer = np.append(self.X_layer, np.zeros(number))
        self.B_layer = np.append(self.B_layer, np.random.uniform(0, 0.00001, number))
        self.A_layer = np.append(self.A_layer, np.zeros(number))
        self.VtB_layer = np.append(self.VtB_layer, np.zeros(number))
        self.MtB_layer = np.append(self.MtB_layer, np.zeros(number))
        self.gE_layer = np.append(self.gE_layer, np.zeros(number))
        self.EQ_layer = np.append(self.EQ_layer, active_fn)
        self.DE_layer = np.append(self.DE_layer, active_fn)
        self.Layer_shape = np.append(self.Layer_shape, number)

    def sym_simplify_eq(self):
        for i, val in enumerate(self.EQ_layer):
            sym.Symbol('x')
            self.EQ_layer[i] = sym.simplify(val)
            self.DE_layer[i] = sym.simplify(diff(val))
        self.interpreting = True

    #LOSS FUNCTIONS
    def MPE(self):
        """
        Mean Absolute Percentage Error
        """
        x = self.output
        y = self.target_val
        a = np.empty(0)
        for i,j in zip(x,y):
            if j != 0 and i != j:
                a = np.append(a, (i - j) / (j * np.abs(i - j)))
            else:
                a = np.append(a, 0)
        self.gE_layer[-self.Layer_shape[-1]:] = a
        if y != 0:
            self.error = 100 * np.mean( np.abs((y-x)/y) )
        else:
            print('MPE gonna 0 divide check the program')
            exit()

    def MSE(self):
        """
        Mean Square Error
        """
        # the constant value of power doesn't consider since the iteration will be convergence to the average value
        # Therefore -> dE_dA = -(target - out)
        self.gE_layer[-self.Layer_shape[-1]:] = -(self.target_val - self.output)
        self.error = np.mean((self.target_val - self.output)** 2)

    def RMSE(self):
        """
        Root Mean Squared Error
        """
        self.gE_layer[-self.Layer_shape[-1]:] = -(self.target_val - self.output)
        self.error = np.sqrt(np.mean((self.target_val - self.output)**2))

    def CROSS_ENTROPY(self):
        self.gE_layer[-self.Layer_shape[-1]:] = self.target_val/self.output
        self.error = -np.sum(self.target_val*np.log(self.output))

    def BINARY_CROSS(self):
        """
        Binary Cross Entropy
        """
        self.gE_layer[-self.Layer_shape[-1]:]  = \
            -(  #y * log(a)
                self.target_val * np.log(self.output)
                #+(1-y)
                + (np.ones(self.Layer_shape[-1]) - self.target_val)
                #*log(1-a)
                * np.log(np.ones(self.Layer_shape[-1]) -self.output)
              )
        self.error = -sum(self.target_val * np.log(self.output))

    def RELATIVE_ENTROPY(self):
        """
        Relative Entropy Error
        """
        self.gE_layer[-self.Layer_shape[-1]:]  = \
            -self.target_val/self.output
        self.error = np.sum(self.target_val * np.log(self.target_val / self.output))

    def show(self, detail = False):
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f' == = == input == = == ')
        print(f'{self.A_layer[0:self.Layer_shape[0]]}')
        if detail is True:
            j = 0
            print('\n == = == W layer == = == ')
            for k in range(len(self.Layer_shape) - 1):
                print(f'layer{k}\n', self.W_layer[j: j + self.Layer_shape[k] * self.Layer_shape[k + 1]].reshape(self.Layer_shape[k], self.Layer_shape[k + 1]))
                j += self.Layer_shape[k] * self.Layer_shape[k + 1]
            j = 0
            print('\n == = == B layer == = == ')
            for k in self.Layer_shape:
                print(f'layer{k}\n', self.B_layer[j: j + k])
                j += k
            j = 0
            print('\n == = == Z layer == = == ')
            for k in self.Layer_shape:
                print(f'layer{k}\n', self.Z_layer[j: j + k])
                j += k
            j = 0
            print('\n == = == A layer == = == ')
            for k in self.Layer_shape:
                print(f'layer{k}\n', self.A_layer[j: j + k])
                j += k
        print('\n == = == Layer Shape == = == ')
        print(self.Layer_shape)
        print(' == = == RESULT == = == ')
        print(f' Target : {self.target_val}')
        print(f' OutPut : {self.output}')
        print(f' Each_error : {self.gE_layer[-self.Layer_shape[-1]:] * -1}')
        print(f' Error : {self.error * 100:.20f} %')
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>')

