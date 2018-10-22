import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 100  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.02 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  return np.exp(x[:])/np.sum(np.exp(x))

def category(x):
  temp = 0
  for i in range(len(x)):
    if x[i] > x[temp]:
      temp = i
  return temp

def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  file = open(fname, "rb")
  temp = pickle.load(file)
  images = temp[:,:784]
  labels = np.zeros((temp.shape[0], 10))
  for i in range(temp.shape[0]):
    labels[i][int(temp[i][784])] = 1
  file.close()
  return images, labels


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.ReLU(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return 1.0/(1.0 + np.exp((-1.0)*x))

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.tanh(x)

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    temp = []
    for i in range(self.x[0].shape[0]):
      if x[0][i] <= 0:
        temp.append(0)
      else:
        temp.append(x[0][i])
    return np.array(temp)

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    return self.sigmoid(self.x)*(1 - self.sigmoid(self.x))

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    return 1 - np.power(self.tanh(self.x),2)

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    temp = []
    for i in range(self.x[0].shape[0]):
      if self.x[0][i] <= 0:
        temp.append(0)
      else:
        temp.append(1)
    return np.array(temp)


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = (np.matmul(x, self.w) + self.b)[0]
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_x = np.array([np.sum(delta*self.w[i]) for i in range(self.w.shape[0])])
    self.d_w = np.outer(self.x, delta)
    self.d_b = delta[:]
    return self.d_x
  
  def update_weights(self, alpha):
    self.w = self.w + alpha*self.d_w
    self.b = self.b + alpha*self.d_b

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets = targets
    temp = x
    for i in self.layers:
      temp = i.forward_pass(temp)
    self.y = softmax(temp)
    if type(targets) == None:
      loss = None
    else:
      loss = self.loss_func(self.y, targets)
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    return (-1.0)*np.dot(targets, np.log(logits + 0.0000000001))
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    delta = self.targets - self.y
    for i in range(len(self.layers)-1, -1, -1):
      delta = self.layers[i].backward_pass(delta)
  
  def update_weights(self, alpha):
    for i in self.layers:
      if type(i) == Layer:
        i.update_weights(alpha)
    
    


      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  arr_shuffle = []
  for i in range(0, X_train.shape[0]):
    arr_shuffle.append(i)
  
  acc_list = []

  for e in range(config["epochs"]):
    print("EPOCH: ", e)
    best_loss = -1.0
    tot_loss = 0.0
    random.shuffle(arr_shuffle)
    for i in range(config["batch_size"]):
      #print(i)
      x = X_train[arr_shuffle[i]]
      t = y_train[arr_shuffle[i]]
      loss, a = model.forward_pass(x, t)
      tot_loss += loss
      if (best_loss > 0 and tot_loss > best_loss):
        break
      model.backward_pass()
      model.update_weights(config["learning_rate"])
    print(tot_loss)
    if best_loss < 0 or tot_loss < best_loss:
      best_loss = tot_loss
    acc = test(model, X_valid, y_valid, config)
    acc_list.append(acc)
    print(acc)
  plt.plot([i for i in range(0,len(acc_list))], acc_list)
  plt.show()
  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  total = X_test.shape[0]
  wrong = 0.0
  for i in range(X_test.shape[0]):
    loss, a = model.forward_pass(X_test[i], y_test[i])
    t = y_test[i]
    if category(a) != category(t):
      wrong += 1
  return wrong/total
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)

