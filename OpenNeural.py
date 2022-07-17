import random
import time
import sympy as sym
from sympy import sqrt, diff
import numpy as np

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
    Layer_shape = np.empty(0, dtype = np.int32) # it contains Layer shapes information
    VtW_layer = np.empty(0, dtype = np.double) # Velocity layer by W for RMSP
    MtW_layer = np.empty(0, dtype = np.double) # Momentum layer by W for Adam
    VtB_layer = np.empty(0, dtype = np.double) # as same as above but by B
    MtB_layer = np.empty(0, dtype = np.double) # as same as above but by B
    Er_layer = np.empty(0, dtype = np.double) # It contains the error for each result and A layer
    target_val = np.empty(0, dtype = np.double) # It is target value, the size is as same as the last value of the layer shape
    soft_on = False # using for soft max in last layer
    normal_on = False # using for norma min max in last layer
    drop_out = 0.0 # drop out rate
    error = 1000 # loss function's error
    learning_rate = 0.01
    interpreting = False # it is using for when forward and backward, do the sympy's equation calculating or not
    iteration_time = 0 # For using the AdamRMSP iteration value
    beta_1 = 0.9 # For using velocity rate
    beta_2 = 0.9 # For using momentum rate
    epsilon = 0.00000001 # For using velocity rate (to prevent dividing by 0)
    adamRMSP = True
    def cpu_run(self):
        """
       It is calculating forward propagation and save each values in Z,A layers\n
       The output should be saved in last layer's A layer\n
       If interpreting is True then interpreting equation as user input at add layer function\n
       It must take much longer than default function leakReLU\n
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
                if self.EQ_layer[i] == sym.simplify('softmax'):  # normalize the output value
                    self.A_layer[a_next: a_next + a_shape] = softmax(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('max_min_lmit'):  # normalize the values between -1, 1, always the max = 1 and min = -1
                    self.A_layer[a_next: a_next + a_shape] = max_min_lmit(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('normal'):  # normalize the output value without consider variance
                    self.A_layer[a_next: a_next + a_shape] = normalminmax(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('znormal'):  # normalize the output value consider variance within -1, 1
                    self.A_layer[a_next: a_next + a_shape] = znormal(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('sigmoid'):  # The typical activation function make between 0 1
                    self.A_layer[a_next: a_next + a_shape] = sigmoid(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('tanh(x)'):  # The typical -1, 1 activation function
                    self.A_layer[a_next: a_next + a_shape] = HyperBolic(
                        self.Z_layer[a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('ReLU'):  # Fast and prevent gradient vanishing
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0, self.Z_layer[
                                                                                               a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify('leakReLU'):  # prevent dead neuron
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(0.3 * self.Z_layer[
                                                                                               a_next: a_next + a_shape],
                                                                                             self.Z_layer[
                                                                                               a_next: a_next + a_shape])
                elif self.EQ_layer[i] == sym.simplify(
                        'parametricReLU'):  # prevent dead neuron and also negative flliping
                    self.A_layer[a_next: a_next + a_shape] = np.maximum(-0.3 * self.Z_layer[
                                                                                               a_next: a_next + a_shape],
                                                                                             self.Z_layer[
                                                                                               a_next: a_next + a_shape])
                else:
                    self.A_layer[a_next: a_next + a_shape] = self.Z_layer[
                                                                                 a_next: a_next + a_shape]
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

    def cpu_back_propagation(self):
            """
        Back propagation, if the interpreting is False then the dA_dZ is based on the derivative of soft ReLU (default)\n
        If not, it follows the derivative function by sympy (it makes the delay of calculating)
         """
        a_next = len(self.A_layer) - self.Layer_shape[-1]
        w_next = len(self.W_layer) - self.Layer_shape[-1] * self.Layer_shape[-2]
        for i in reversed(range(1, len(self.Layer_shape))):
            a_shape = self.Layer_shape[i]
            w_shape = self.Layer_shape[i] * self.Layer_shape[i - 1]
            dE_dA = self.Er_layer[a_next: a_next + a_shape]
            # calculate partial derivative of A
            dA_dX = np.ones(a_shape)  # typical x
            if self.interpreting:
                dA_dX = np.empty(0)
                for k in range(a_shape):
                    v_x = sym.Symbol('x')
                    val = self.Z_layer[a_next + k]
                    dA_dX = np.append(dA_dX, float(self.DE_layer[i].subs(v_x, val).evalf()))
            if self.EQ_layer[i] == sym.simplify('sigmoid'):
                dA_dX = self.X_layer[a_next: a_next + a_shape] * (1 - self.X_layer[a_next: a_next + a_shape])
            elif self.EQ_layer[i] == sym.simplify('tanh(x)'):
                dA_dX = HyperBolic(self.X_layer[a_next: a_next + a_shape]) ** 2
            elif self.EQ_layer[i] == sym.simplify('ReLU'):
                dA_dX = np.where(self.X_layer[a_next: a_next + a_shape]> 0, 1, 0)
            elif self.EQ_layer[i] == sym.simplify('parametricReLU'):
                dA_dX = np.where(self.X_layer[a_next: a_next + a_shape] > 0, 1, -0.3)
            elif self.EQ_layer[i] == sym.simplify('leakReLU'):
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
                mdw_corr = self.MtW_layer[w_next:w_next + w_shape] / (1 - self.beta_1 ** self.iteration_time)
                vdw_corr = self.VtW_layer[w_next:w_next + w_shape] / (1 - self.beta_2 ** self.iteration_time)
                w_update = self.learning_rate * (mdw_corr/(np.sqrt(vdw_corr) + self.epsilon))
                # AdamRMSP Bias
                self.MtB_layer[a_next: a_next + a_shape] = \
                    self.beta_1 * self.MtB_layer[a_next: a_next + a_shape] + (1 - self.beta_1) * dE_dZ
                self.VtB_layer[a_next : a_next + a_shape] =\
                    self.beta_2 * self.VtB_layer[a_next : a_next + a_shape] + (1 - self.beta_2) * dE_dZ**2
                vdb_corr = self.VtB_layer[a_next : a_next + a_shape] / (1 - self.beta_2 ** self.iteration_time)
                mdb_corr = self.MtB_layer[a_next : a_next + a_shape] / (1 - self.beta_1 ** self.iteration_time)
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
            self.Er_layer[a_next - self.Layer_shape[i - 1]: a_next] =\
                np.matmul(self.W_layer[w_next:w_next + w_shape].reshape(self.Layer_shape[i-1],self.Layer_shape[i]),
                          np.transpose(dE_dZ) ).flatten() #transpose multiplication
            #next iteration
            a_next -= self.Layer_shape[i - 1]
            w_next -= self.Layer_shape[i - 1] * self.Layer_shape[i - 2]
        self.iteration_time += 1

    def cl_run(self):
        """
        It is calculating forward propagation and save each values in Z,A layers\n
        The output should be saved in last layer's A layer\n
        If interpreting is True then interpreting equation as user input at add layer function\n
        It must take much longer than default function leakReLU\n
        """
        pass # too lazy to do it again I will do it later

    def cl_back_propagation(self):
        """
        Back propagation, if the interpreting is False then the dA_dZ is based on the derivative of soft ReLU (default)\n
        If not, it follows the derivative function by sympy (it makes the delay of calculating)
        """
        pass # too lazy to do it again I will do it later

    def csv_save(self, file_name):
        """
        It only exports the W_layer for current neural network\n
        :param file_name: it is file name for export the file should be written as "file_name.csv"
        :return:
        """
        np.savetxt(file_name, self.W_layer, delimiter = ',')

    def csv_load(self, file_name):
        """
        It is overlapping the W layer of current neural network\n
        The overlapped neural network's W_layer should be generated and also have the same shape\n
        :param file_name: it is file name which is in the same directory with this python script
        """
        self.W_layer = np.loadtxt(file_name, delimiter = ',')

    def run_init(self, input_val, dropout = 0.0, interpreting = False):
        """
        :param input_val: it is the input value which will be contained in the first A_layer
        :param dropout: it is the drop out rate between [0~1], recommend 0 when doing reinforcement learnin
        :return:
        """
        # initializing
        self.Er_layer = np.zeros(self.Er_layer.size)
        self.VtW_layer = np.zeros(self.VtW_layer.size)
        self.MtW_layer = np.zeros(self.MtW_layer.size)
        self.VtB_layer = np.zeros(self.VtB_layer.size)
        self.MtB_layer = np.zeros(self.MtB_layer.size)
        self.Z_layer = np.zeros(self.Z_layer.size)
        self.X_layer = np.zeros(self.X_layer.size)
        self.A_layer = np.zeros(self.A_layer.size)
        self.Z_layer[0:self.Layer_shape[0]] = np.array(input_val)[0:self.Layer_shape[0]] #input
        self.drop_out = dropout
        self.interpreting = interpreting

    def learn_set(self, input_val, target_val, learning_rate  = 0.001, dropout = 0.02, optima = True):
        """
        :param input_val: it is the input value which will be contained in the first A_layer
        :param target_val: it is the target value which will be using for the measure cost value
        :param learning_rate: it is the learning rate for back propgation
        :param dropout: it is the drop out rate between [0~1], recommend 0 when doing reinforcement learning
        :return:
        """
        # initializing
        self.run_init(input_val, dropout = dropout)
        self.error = 100
        self.learning_rate = learning_rate
        self.iteration_time = 1
        self.target_val = np.array(target_val)[0:self.Layer_shape[-1]] #target

    def learn_start(self, max_trial = 300, accruancy = 0.01, loss_fun = 'RMSE', DQN_ACT = 0, show_result = True):
        """
        :param max_trial: it is the maximum trials for backpropagation
        :param accruancy: it decides the break point for the allowed minimum error
        :param loss_fun: MSE, RMSE, MPE, BINARY_CROSS, RELATIVE_ENTROPY, HUBER_LOSS  default : MSE
        :param show_result: True or False for showing the result when the learning is terminated
        :return:
        """
        copy_W_layer = np.copy(self.W_layer)
        start = time.time()
        self.iteration_time = 1
        while True:
            self.cpu_run()
            if loss_fun == 'RMSE':
                self.RMSE()
            elif loss_fun == 'MPE':
                self.MPE()
            elif loss_fun == 'BINARY_CROSS':
                self.BINARY_CROSS()
            elif loss_fun == 'RELATIVE_ENTROPY':
                self.RELATIVE_ENTROPY()
            elif loss_fun == 'TD_LOSS':
                self.TD_LOSS(DQN_ACT)
            elif loss_fun == 'HUBER_LOSS':
                self.HUBER_LOSS(DQN_ACT)
            else:
                self.MSE()
            if self.iteration_time >= max_trial or self.error < accruancy:
                break
            self.cpu_back_propagation()
            if show_result:
                print(self.iteration_time, self.error)
            if np.any(np.isnan(self.W_layer)) \
                    or np.any(np.isinf(self.W_layer)) \
                    or np.any(np.isnan(self.A_layer)) \
                    or np.any(np.isinf(self.A_layer)) \
                    or np.any(np.isnan(self.VtW_layer))\
                    or np.any(np.isinf(self.VtW_layer)):
                self.W_layer = copy_W_layer
                if show_result:
                    self.learn_show('Cyan', time.time() - start)
                break
        end = time.time()
        if self.iteration_time >= max_trial:
            if show_result:
                self.learn_show('Red', end - start)
        elif self.error < accruancy :
            if show_result:
                self.learn_show('Green', end - start)

    def learn_show(self, color, time):
        np.set_printoptions(precision = 4)
        if color == 'Green':
            print(Fore.GREEN + f'[save] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.iteration_time} '
                    f'trg :{self.target_val[0:self.Layer_shape[-1]]}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val[0:self.Layer_shape[-1]] - self.A_layer[-self.Layer_shape[-1]:], 2)} '
                    f'err :{self.error:.4f}% '
                    f'tim :{time :.2f}s'
                    + Fore.RESET)
        elif color == 'Red':
            print(Fore.RED + f'[save] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.iteration_time} '
                    f'trg :{self.target_val[0:self.Layer_shape[-1]]}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val[0:self.Layer_shape[-1]] - self.A_layer[-self.Layer_shape[-1]:], 2)} '
                    f'err :{self.error:.4f}% '
                    f'tim :{time :.2f}s'
                    + Fore.RESET)
        elif color == 'Cyan':
            print(Fore.CYAN + f'[error] ',
                    f'inp :{self.A_layer[0:self.Layer_shape[0]]}',
                    f'tri :{self.iteration_time} '
                    f'trg :{self.target_val[0:self.Layer_shape[-1]]}',
                    f'out :{self.A_layer[-self.Layer_shape[-1]:]}',
                    f'ech :{np.power(self.target_val[0:self.Layer_shape[-1]] - self.A_layer[-self.Layer_shape[-1]:], 2)} '
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
        :param active_fn: softmax, max_min_lmit, normal, znormal, sigmoid, tanh(x), ReLU, leakReLU, parametricReLU if not set then it works aas 'x'
        :return:
        """
        self.Z_layer = np.append(self.Z_layer, np.zeros(number))
        self.X_layer = np.append(self.X_layer, np.zeros(number))
        self.B_layer = np.append(self.B_layer, np.random.uniform(0, 0.00001, number))
        self.A_layer = np.append(self.A_layer, np.zeros(number))
        self.VtB_layer = np.append(self.VtB_layer, np.zeros(number))
        self.MtB_layer = np.append(self.MtB_layer, np.zeros(number))
        self.Er_layer = np.append(self.Er_layer, np.zeros(number))
        self.EQ_layer = np.append(self.EQ_layer, sym.simplify(active_fn))
        self.DE_layer = np.append(self.DE_layer, sym.simplify(diff(active_fn)))
        self.Layer_shape = np.append(self.Layer_shape, number)
    #LOSS FUNCTIONS
    def MPE(self):
        """
        Mean Absolute Percentage Error
        """
        x = self.A_layer[-self.Layer_shape[-1]:]
        y = self.target_val[0:self.Layer_shape[-1]]
        a = np.empty(0)
        for i,j in zip(x,y):
            if j != 0 and i != j:
                a = np.append(a, (i - j) / (j * np.abs(i - j)))
            else:
                a = np.append(a, 0)
        self.Er_layer[-self.Layer_shape[-1]:] = a
        self.error = 100 * np.mean( np.abs((y-x)/y) )

    def MSE(self):
        """
        Mean Square Error
        """
        # the constant value of power doesn't consider since the iteration will be convergence to the average value
        # Therefore -> dE_dA = -(target - out)
        self.Er_layer[-self.Layer_shape[-1]:] = -(self.target_val - self.A_layer[-self.Layer_shape[-1]:])
        self.error = np.mean((self.target_val - self.A_layer[-self.Layer_shape[-1]:])** 2)

    def RMSE(self):
        """
        Root Mean Squared Error
        """
        self.Er_layer[-self.Layer_shape[-1]:] = -(self.target_val - self.A_layer[-self.Layer_shape[-1]:])
        self.error = np.sqrt(np.mean((self.target_val[0:self.Layer_shape[-1]] - self.A_layer[-self.Layer_shape[-1]:])**2))

    def BINARY_CROSS(self):
        """
        Binary Cross Entropy
        """
        self.Er_layer[-self.Layer_shape[-1]:]  = \
            -(  #y * log(a)
                self.target_val[0:self.Layer_shape[-1]] * np.log(self.A_layer[-self.Layer_shape[-1]:])
                #+(1-y)
                + (np.ones(self.Layer_shape[-1]) - self.target_val[0:self.Layer_shape[-1]])
                #*log(1-a)
                * np.log(np.ones(self.Layer_shape[-1]) -self.A_layer[-self.Layer_shape[-1]:])
              )
        self.error = -sum(self.target_val[0:self.Layer_shape[-1]] * np.log(self.A_layer[-self.Layer_shape[-1]:]))

    def RELATIVE_ENTROPY(self):
        """
        Relative Entropy Error
        """
        self.Er_layer[-self.Layer_shape[-1]:]  = \
            -self.target_val[0:self.Layer_shape[-1]]/self.A_layer[-self.Layer_shape[-1]:]
        self.error = np.sum(self.target_val[0:self.Layer_shape[-1]] * np.log(self.target_val[0:self.Layer_shape[-1]] / self.A_layer[-self.Layer_shape[-1]:]))

    def TD_LOSS(self, ACT):
        self.Er_layer[-self.Layer_shape[-1]:] = (self.target_val - self.A_layer[-self.Layer_shape[-1]:])
        self.error = np.mean((self.target_val - self.A_layer[-self.Layer_shape[-1]:])** 2)

    def HUBER_LOSS(self, ACT):
        """
        Huber Loss for using DQN Loss
        """
        self.Er_layer[-self.Layer_shape[-1]:] = self.target_val[0:self.Layer_shape[-1]] - self.A_layer[-self.Layer_shape[-1]:]
        y = self.target_val[0:self.Layer_shape[-1]][ACT]
        x = self.A_layer[-self.Layer_shape[-1]:][ACT]
        a = y - x
        if np.abs(a) < a**2 :
            self.error = 0.5 * a ** 2
            self.Er_layer[-self.Layer_shape[-1]:][ACT] = -2 * (y-x)
        else:
            self.error = a**2 * (np.abs(a)-0.5*a**2)
            self.Er_layer[-self.Layer_shape[-1]:][ACT] = a*(x-y)/np.abs(x-y)

    def show(self, detail):
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
        print(f' Target : {self.target_val[0:self.Layer_shape[-1]]}')
        print(f' OutPut : {self.A_layer[-self.Layer_shape[-1]:]}')
        print(f' Each_error : {self.Er_layer[-self.Layer_shape[-1]:] * -1}')
        print(f' Error : {self.error * 100:.20f} %')
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>')
