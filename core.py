###############################################################################
# Import required packages
import os
import numpy as np
import time
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
###############################################################################

###############################################################################
# Function that prepares TensorFlow
def prepare_tensorflow():
    
    # Reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    
    # Check for GPU computing
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print ("GPU computing is available")
    else:
        print ("GPU computing is NOT available")
        
    # Check tf version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
###############################################################################

###############################################################################
# Function that predicts stochastic colloidal self-assembly dynamics
# Note that "states" input can have shape (nx+nw, 1) or (nx+nw,)
def stoch_dyn_CSA(states):
    
    # Distribute states
    xk = states[0]
    uk = states[1]
    xkw = states[2]
    
    # Enter parameters
    Kb = 1.38064852 * 10**-23 # (J/K)
    T = 293 # (K)
    dt = 1 # (s) -- sampling time
    
    # Get diffusion coefficient
    g2 = 0.0045*np.exp(-(xk-2.1-0.75*uk)**2)+0.005 # Diffusion coefficient
    
    # Get drift coefficient
    #    F = 10*K*T*(x-2.1-0.75*u)**2
    dFdx = 20*Kb*T*(xk-2.1-0.75*uk)
    dg2dx = -2*(xk-2.1-0.75*uk)*0.0045*np.exp(-(xk-2.1-0.75*uk)**2)
    g1 = -(g2*dFdx/(Kb*T)-dg2dx)
    
    # Predict forward dynamics
    xkp1 = xk + g1*dt + np.sqrt(2*g2*dt)*xkw
    
    return [np.asarray([xkp1]), np.asarray([g1]), np.asarray([g2])]
###############################################################################

###############################################################################
# Function that predicts stochastic competitive Lotka-Volterra dynamics with
# coexistence equilibrium
# Note that "states" input can have shape (nx+nw, 1) or (nx+nw,)
def stoch_dyn_LVE(states):
    
    # Distribute states
    xk = states[0]
    yk = states[1]
    xkw = states[2]
    ykw = states[3]
    
    # Enter parameters
    k1 = 0.4
    k2 = 0.5
    xeq = 0.75
    yeq = 0.625
    d1 = 0.5
    d2 = 0.5
    dt = 0.01
    
    # Get drift coefficients
    g1x = xk*(1 - xk - k1*yk)
    g1y = yk*(1 - yk - k2*xk)
    
    # Get diffusion coefficients
    g2x = 1/2*(d1*xk*(yk-yeq))**2
    g2y = 1/2*(d2*yk*(xk-xeq))**2
    
    # Predict forward dynamics
    xkp1 = xk + g1x*dt + np.sqrt(2*g2x*dt)*xkw
    ykp1 = yk + g1y*dt + np.sqrt(2*g2y*dt)*ykw
    
    return [np.asarray([[xkp1], [ykp1]]), np.asarray([[g1x], [g1y]]), 
            np.asarray([[g2x], [g2y]])]
###############################################################################

###############################################################################
# Function that predicts stochastic SIR dynamics
# Note that "states" input can have shape (nx+nw, 1) or (nx+nw,)
def stoch_dyn_SIR(states):
    
    # Distribute states
    sk = states[0]
    ik = states[1]
    rk = states[2]
    skw = states[3]
    ikw = states[4]
    rkw = states[5]
    
    # Enter parameters
    b = 1
    d = 0.1
    k = 0.2
    alpha = 0.5
    gamma = 0.01
    mu = 0.05
    h = 2
    delta = 0.01
    sigma_1 = 0.2
    sigma_2 = 0.2
    sigma_3 = 0.1
    dt = 1
    
    # Get nonlinear incidence rate
    g = (k*sk**h*ik)/(sk**h+alpha*ik**h)
    
    # Get drift coefficients
    g1s = b-d*sk-g+gamma*rk
    g1i = g-(d+mu+delta)*ik
    g1r = mu*ik-(d+gamma)*rk
    
    # Get diffusion coefficients
    g2s = 1/2*(sigma_1*sk)**2
    g2i = 1/2*(sigma_2*ik)**2
    g2r = 1/2*(sigma_3*rk)**2
    
    # Predict forward dynamics
    skp1 = sk + g1s*dt + np.sqrt(2*g2s*dt)*skw
    ikp1 = ik + g1i*dt + np.sqrt(2*g2i*dt)*ikw
    rkp1 = rk + g1r*dt + np.sqrt(2*g2r*dt)*rkw
    
    return [np.asarray([skp1, ikp1, rkp1]), np.asarray([g1s, g1i, g1r]), 
            np.asarray([g2s, g2i, g2r])]

###############################################################################

###############################################################################
# Function that creates Unscented Transform (UT) scaling parameters and weights
def get_weights(nx, nw):
    
    # Enter UT parameters (standard)
    beta = 0
    alpha = 1
    kappa = 0
    
    # Get total system size
    n = nx + nw
    
    # Get lambda
    lam = alpha**2*(n+kappa)-n
    
    # Get weights
    Wm = np.zeros((2*n+1,1)) # mean weight vector
    Wc = np.zeros((2*n+1,1)) # covariance weight vector
    Wm[0] =lam/(n+lam)
    Wc[0] = lam/(n+lam) + ((1-alpha**2+beta))
    Wm[1:2*n+1,:] = 1/(2*(n+lam))*np.ones((2*n,1))
    Wc[1:2*n+1,:] = 1/(2*(n+lam))*np.ones((2*n,1))
    
    # Normalize weights
    Wm = Wm/np.sum(Wm)
    Wc = Wc/np.sum(Wc)
    
    # Ensure float 32 data type
    return np.float32(lam), np.float32(Wm), np.float32(Wc)
###############################################################################

###############################################################################
# Function that creates sigma point matrix for UT
# Note that we create sigma point matrices whose columns are in the following
# order: [x, u, w]^T
# Note that the mean input needs to be of shape (nx+nu, 1) and the variance
# input needs to be of shape (nx, nx)
def get_sigma(mean, variance, nx, nu, nw, lam):
    
    # Get total system size
    n = nx + nw
    
    # Combine variances of states and process noise variables into n x n matrix
    SS = block_diag(variance, np.identity(nw))
    
    # Take square root of this matrix (multiplied by (n+lam))
    S = sqrtm((n+lam)*SS)
    
    # Conatenate state mean with means of process noise variables
    concat_mean = np.concatenate((mean[0:nx,:], np.zeros((nw,1))), axis=0)
    
    # Initialize matrix of sigma points with mean values for every entry
    chi = concat_mean*np.ones((n, 2*n+1))
    
    # Adjust sigma points based on variance
    count = 0
    for i in range(1, n+1):
        chi[:,i] = chi[:,i] + S[:,count]
        count = count + 1
    
    count = 0
    for j in range(n+1,2*n+1):
        chi[:,j] = chi[:,j] - S[:,count]
        count = count + 1
    
    # Insert row(s) for exogenous input(s) (if exogenous inputs exist)
    if nu > 0:
        for i in range(0, nu):
            chi = np.insert(chi, nx+i, mean[nx+i,0]*np.ones(2*n+1), 0)
    
    return np.float32(chi)
###############################################################################

###############################################################################
# Function that performs (entire) UT propagation
# Note that the mean input needs to be of shape (nx+nu, 1) and the variance
# input needs to be of shape (nx, nx)
def UT(stoch_dyn, mean, variance, nx, nu, nw):
    
    # Get total system size
    n = nx+nw
    
    # Get weights and scaling parameters
    lam, Wm, Wc =  get_weights(nx, nw)
    
    # Get sigma points
    chi = get_sigma(mean, variance, nx, nu, nw, lam)
    
    # Propagate sigma points through stochastic dynamics
    y = np.zeros((nx, 2*n+1))
    for i in range(0, 2*n+1):
        y[:,i], _, _ = stoch_dyn(chi[:,i])
    
    # Get predicted mean and variance
    mean = np.matmul(y, Wm)
    
    variance = np.matmul(Wc.T*(y-mean), np.transpose((y-mean)))
    
    return mean, variance, y, chi
###############################################################################

###############################################################################
# Function that standardizes data
def standardize(X, mu, std):
    return (X-mu)/std

# Function that "un-standardizes" data
def un_standardize(X, mu, std):
    return (X*std)+mu

# Function that finds mean and standard deviation of data set
def find_mu_std(X):
    
    # Pre-allocate
    std = np.zeros((np.shape(X)[1]))
    mu = np.zeros((np.shape(X)[1]))
    
    # Find mu, std
    for i in range(0, np.shape(X)[1]):
        std[i] = np.std(X[:,i])
        mu[i] = np.mean(X[:,i])
    
    return mu, std
###############################################################################
    
###############################################################################
# Function that splits data into training, validation, and testing sets
def train_val_test_split(X, Y):
    
    TF = 0.60 # fraction of data used for training
    
    X_train, X_val_test, Y_train, Y_val_test  = train_test_split(X, Y, 
                                                                 test_size=(1-TF))
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test,
                                                    Y_val_test, test_size=0.50)
    
    return [X_train, X_val, X_test, Y_train, Y_val, Y_test]
###############################################################################

###############################################################################
# Function that preps training data for estimating drift coefficient as a
# neural network
# Note that the function uses UT for uncertainty propagation
# Note that mean_initial/mean_final must have shape(nx+nu, 1) while
# cov_initial/cov_final must have shape (nx, nx)
def g1_train_prep(mean_initial, mean_final, cov_initial, nx, nu, nw, lam, dt, 
                  g1_path):
    
    # Get total system size
    n = nx + nw
    
    # Initialize
    X = [] # [x(k), u(k)] where k is discrete time point (if u exists)
    Y = [] # ((X(k+1))-X(k))/dt
    CHI = [] # Sigma points

    for i in range(0, np.shape(mean_initial)[0]):
        
        # Get initial and final mean states
        xk = mean_initial[i,:,0]
        xkp1 = mean_final[i,:,0]
        
        # Get sigma points
        chi = get_sigma(mean_initial[i,:,:], cov_initial[i,:,:], nx, nu, nw, lam)
        
        # Record initial states
        X.append(np.array(xk))
    
        # Record final states
        Y.append((xkp1-xk[0:nx])/dt)
        
        # Record relevant sigma points (except for the mean)
        CHI.append(chi[0:nx+nu,1:].transpose().flatten())
    
    # Convert to array
    X = np.asarray(X)
    Y = np.asarray(Y)
    CHI = np.asarray(CHI)
    
    # Concatenate sigma points to Y so that they are easier to deal with
    # when training neural network with keras
    Y_concat = np.concatenate((Y, CHI), axis=1)
    
    # Split into training, validation, and testing data
    [X_train, X_val, X_test, 
     Y_train, Y_val, Y_test] = train_val_test_split(X, Y_concat)
    
    # Find mean and standard deviation of neural network input based on 
    # training data
    X_mu, X_std = find_mu_std(X_train)
    
    # Standardize neural network input
    X_train_scaled = standardize(X_train, X_mu, X_std)
    X_val_scaled = standardize(X_val, X_mu, X_std)
    X_test_scaled = standardize(X_test, X_mu, X_std)
    
    # Find min and max of neural network output, which
    # is the output of the drift coefficient (i.e., Y[:,0:nx])
    Y_mu, Y_std = find_mu_std(Y_train[:,0:nx])
    
    # Pre-allocate
    Y_train_scaled = np.zeros(np.shape(Y_train))
    Y_val_scaled = np.zeros(np.shape(Y_val))
    Y_test_scaled = np.zeros(np.shape(Y_test))

    # Standardize Y[:,0:nx]
    Y_train_scaled[:,0:nx] = standardize(Y_train[:,0:nx], Y_mu, Y_std)
    Y_val_scaled[:,0:nx] = standardize(Y_val[:,0:nx], Y_mu, Y_std)
    Y_test_scaled[:,0:nx] = standardize(Y_test[:,0:nx], Y_mu, Y_std)
    
    
    # Normalize sigma points (i.e., remaining columns of Y)
    for i in range(0, nx+nu):
        for j in range(0, 2*n):
            Y_train_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_train[:,nx+i+j*(nx+nu)],
                                                      X_mu[i], X_std[i])
            Y_val_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_val[:,nx+i+j*(nx+nu)], 
                                                    X_mu[i], X_std[i])
            Y_test_scaled[:,nx+i+j*(nx+nu)] = standardize(Y_test[:,nx+i+j*(nx+nu)], 
                                                     X_mu[i], X_std[i])
        
    
    # Ensure float32 data type and save
    X_train_scaled = np.float32(X_train_scaled)
    X_val_scaled = np.float32(X_val_scaled)
    X_test_scaled = np.float32(X_test_scaled)
    Y_train_scaled = np.float32(Y_train_scaled)
    Y_val_scaled = np.float32(Y_val_scaled)
    Y_test_scaled = np.float32(Y_test_scaled)
    X_mu = np.float32(X_mu)
    X_std = np.float32(X_std)
    Y_mu = np.float32(Y_mu)
    Y_std = np.float32(Y_std)

    np.save(os.path.join(g1_path, 'X_train_g1.npy'), X_train_scaled)
    np.save(os.path.join(g1_path, 'X_val_g1.npy'), X_val_scaled)
    np.save(os.path.join(g1_path, 'X_test_g1.npy'), X_test_scaled)
    np.save(os.path.join(g1_path, 'Y_train_g1.npy'), Y_train_scaled)
    np.save(os.path.join(g1_path, 'Y_val_g1.npy'), Y_val_scaled)
    np.save(os.path.join(g1_path, 'Y_test_g1.npy'), Y_test_scaled)
    np.save(os.path.join(g1_path, 'X_mu_g1.npy'), X_mu)
    np.save(os.path.join(g1_path, 'X_std_g1.npy'), X_std)
    np.save(os.path.join(g1_path, 'Y_mu_g1.npy'), Y_mu)
    np.save(os.path.join(g1_path, 'Y_std_g1.npy'), Y_std)
    
    return [X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, 
        Y_val_scaled, Y_test_scaled, X_mu, X_std, Y_mu, Y_std]
        
###############################################################################  
     
###############################################################################
# Function that creates neural network based on various network size parameters
def create_NN(input_dim, output_dim, n_hidden_nodes, n_hidden_layers):
    
    # Create first layer
    NN_initial = Dense(n_hidden_nodes, activation = "swish", 
                            input_shape=(input_dim,))
    
    # Create final layer
    NN_final = Dense(output_dim, activation='linear')
    
    # Create hidden layers if # hidden layers >1
    hidden_list = []
    for i in range(0,n_hidden_layers-1):
        hidden_list.append(Dense(n_hidden_nodes, activation = "swish"))
        
    # Create DNN model
    NN = Sequential()
    NN.add(NN_initial)
    for j in range(0, n_hidden_layers-1):
        NN.add(hidden_list[j])
        
    NN.add(NN_final)
    
    return NN
###############################################################################
    
###############################################################################
# Function that compiles neural network
def compile_NN(NN, loss_function):
    learning_rate = 0.0001 # learning rate
    optimizer = keras.optimizers.Adam(lr=learning_rate) # Adam optimizer
    NN.compile(optimizer=optimizer, loss=loss_function) # Compile neural network
    return NN
###############################################################################  

###############################################################################  
# Function that trains single neural network
def train_single_NN(NN, X_train, Y_train, X_val, Y_val):
    n_batch = 32 # batch size
    n_epoch = 10**4 # number of epochs
    es = EarlyStopping(monitor = 'val_loss', 
                           mode = 'min', verbose = 1, 
                           patience = 25) # early stopping

    NN.fit(X_train, Y_train, epochs=n_epoch, batch_size=n_batch,
                shuffle=True, callbacks = [es], 
                validation_data=(X_val, Y_val))
    return NN
###############################################################################  

###############################################################################  
# Function that saves neural network
def save_NN(NN, path):
    
    # Save NN
    model_json_1 = NN.to_json()
    with open(path+"NN.json", "w") as json_file:
        json_file.write(model_json_1)
    NN.save_weights(path+"NN.h5")
###############################################################################  

###############################################################################    
# Function that loads neural network
def load_NN(path):
    
    # Load json and create model
    json_file = open(path + "NN.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_NN = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_NN.load_weights(path + "NN.h5")
    return loaded_NN
###############################################################################

###############################################################################
# Loss function for estimating drift coefficient during training
def g1_train_loss(NN, nx, nu, Wm):
    def UT_mean_prop_loss(y_true, y_pred):
        
        # Output of neural network multiplied by first mean weight is the
        # first sigma point that is propagated through the dynamics. Note that
        # g2 does not contribute becuase its sigma points cancel out (due to 
        # the fact that the mean of w_i = 0)
        yp_total = tf.multiply(Wm[0], y_pred[:,0:nx])
        
        # Propagate remaining sigma points through the dynamics
        for i in range(0, len(Wm)-1):
            yp_total = tf.add(yp_total, tf.multiply(Wm[i+1], NN(y_true[:,(nx+nu)*i+nx:(nx+nu)*i+nx+nu+nx])))
            
        # Calculate squared error between true output of drift coefficient
        # and UT/neural network estimated output
        squared_difference = tf.square(y_true[:,0:nx] - yp_total[:,0:nx])
        
        # Get mean of squared error
        return tf.reduce_mean(squared_difference, axis=-1)
    
    return UT_mean_prop_loss
###############################################################################

###############################################################################
# Loss function for estimating drift coefficient for testing
def g1_test_loss(NN, y_true, x_test, nx, nu, Wm):
    
    # Get predictions of neural network
    y_pred = NN(x_test)
    
    # Output of neural network multiplied by first mean weight is the
    # first sigma point that is propagated through the dynamics. Note that
    # g2 does not contribute becuase its sigma points cancel out (due to 
    # the fact that the mean of w_i = 0)
    yp_total = tf.multiply(Wm[0], y_pred[:,0:nx])
    
    # Propagate remaining sigma points through the dynamics
    for i in range(0, len(Wm)-1):
        yp_total = tf.add(yp_total, tf.multiply(Wm[i+1], NN(y_true[:,(nx+nu)*i+nx:(nx+nu)*i+nx+nu+nx])))
        
    # Calculate squared error between true output of drift coefficient
    # and UT/neural network estimated output
    squared_difference = tf.square(y_true[:,0:nx] - yp_total[:,0:nx])
    
    # Get mean of squared error
    return np.mean(tf.reduce_mean(squared_difference, axis=-1).numpy())
###############################################################################

############################################################################### 
# Function that trains multiple g1 neural networks and saves relevant outputs
def train_multiple_NNs_g1(X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                          n_hidden_layers, n_hidden_nodes, nx, nu, Wm, g1_path):
    
    # Get input and output dimensions of neural network
    input_dim = nx + nu
    output_dim = nx

    # Train neural networks that represent g1
    for nhl in n_hidden_layers:
        for nhn in n_hidden_nodes:
            
            # Create path where model will be saved
            path = os.path.join(g1_path, str(nhl) + "_HL_" + str(nhn) + "_Nodes/")
            os.mkdir(path)
            
            # Create neural network
            NN = create_NN(input_dim, output_dim, nhn, nhl)
            
            # Compile neural network
            NN = compile_NN(NN, g1_train_loss(NN, nx, nu, Wm))
            
            # Train neural network
            start = time.time() # Record start time
            NN = train_single_NN(NN, X_train, Y_train, X_val, Y_val)
            end = time.time() # Record end time
            total_time = end - start
            
            # Save neural network
            save_NN(NN, path)
            
            # Record losses
            train_loss = np.asarray(NN.history.history['loss'])
            val_loss = np.asarray(NN.history.history['val_loss'])
            test_loss = g1_test_loss(NN, Y_test, X_test, nx, nu, Wm)
            
            # Save loss
            np.save(path + "Train_Loss.npy", train_loss)
            np.save(path + "Val_Loss.npy", val_loss)
            np.save(path + "Test_Loss.npy", test_loss)
            
            # Save total run time
            np.save(path + "total_time.npy", total_time)
############################################################################### 

###############################################################################
# Neural network evaluation function (use of tf.function speeds up evaluation)
@tf.function
def predict(NN, vector):
    return NN(vector)
###############################################################################

###############################################################################
# Function that evaluates the deterministic part of the stochastic differential
# equation according to an Euler discretization (i.e., xk + g1(x,u)*dt)
def det_eval(g1_NN, states, X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, dt):
    
    # Reshape input
    states = states.reshape(nx+nu,)
    
    # Scale input
    states_scaled = standardize(states, X_mu_g1, X_std_g1).reshape(1, nx+nu)
    
    # Get predictions from neural network and un-scale them
    g1_outputs_scaled = (predict(g1_NN, states_scaled).numpy()[0]).reshape(nx,)
    g1_outputs = un_standardize(g1_outputs_scaled, Y_mu_g1, Y_std_g1)

    # Include previous state and time discretization
    return states[0:nx] + g1_outputs*dt
        
###############################################################################

###############################################################################
# Function that calculates "target" (i.e., Y) for training diffusion coefficient
# neural network
def g2_target_calc(g1_NN, mean_initial, mean_final, cov_initial, cov_final, 
                   X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam, Wc, 
                   dt):
    
    # Get total system size
    n = nx + nw
     
    # Get sigma points
    chi = get_sigma(mean_initial, cov_initial, nx, nu, nw, lam)
    
    # Get propagated sigma points (assuming g2=0)
    y = np.zeros((nx, 2*n+1))
    for i in range(0, 2*n+1):
        y[:,i] = det_eval(g1_NN, chi[0:nx+nu,i], X_mu_g1, X_std_g1, Y_mu_g1, 
                         Y_std_g1, nx, nu, dt)
    
    # Subtract final mean from y
    y_minus_mean = y-mean_final
    
    # "Zero out" entries that have non-zero g2 contributions
    for i in range(0, nw):
        y_minus_mean[i,1+nx+i] = 0
        y_minus_mean[i,1+nx+n+i] = 0
            
    # Multiply y_minus_mean by covariance weights and square.
    var_pt_1 = np.matmul(Wc.T*(y_minus_mean), np.transpose((y_minus_mean)))
    
    # Use propagated variance
    det_mean = det_eval(g1_NN, mean_initial, X_mu_g1, X_std_g1, Y_mu_g1, 
                        Y_std_g1, nx, nu, dt)
    
    var_pt_2 = np.eye(nx)
    for i in range(0, nx):
        var_pt_2[i,i] = Wc[nx+1]*(2*(det_mean[i])**2 + 2*mean_final[i]**2 - 4*det_mean[i]*mean_final[i])
        
    # Get target
    target = np.array(np.diag(cov_final - var_pt_1 - var_pt_2))
    
    for i in range(0, nw):
       chi_w = chi[nx+nu+i, 1+nx+i]
       target[i] = target[i]/4/Wc[nx+1]/chi_w**2/dt
    
    return target
############################################################################### 
    
###############################################################################
# Function that preps training data for estimating drift coefficient as a
# neural network
def g2_train_prep(g1_NN, mean_initial, mean_final, cov_initial, cov_final, 
                  X_mu_g1, X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam, Wc, 
                  dt, g2_path):
    
    
    # Get data that will be used to train the model
    X = [] # Z(k) where k is discrete time point
    Y = [] # Estimated diffusion coefficient at time k

    for i in range(0, np.shape(mean_initial)[0]):
        
        # Get initial and final mean states
        X.append(mean_initial[i,:,0])
        
        # Get sigma points
        Y.append(g2_target_calc(g1_NN, mean_initial[i,:,:], mean_final[i,:,:], 
                                cov_initial[i,:,:], cov_final[i,:,:], X_mu_g1, 
                                X_std_g1, Y_mu_g1, Y_std_g1, nx, nu, nw, lam,
                                Wc, dt))
    # Convert to array
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Split into training, validation, and testing data
    [X_train, X_val, X_test, 
     Y_train, Y_val, Y_test] = train_val_test_split(X, Y)
     
    # Find mean and standard deviation of neural network input based on 
    # training data
    X_mu, X_std = find_mu_std(X_train)
    
    # Normalize neural network input
    X_train_scaled = standardize(X_train, X_mu, X_std)
    X_val_scaled = standardize(X_val, X_mu, X_std)
    X_test_scaled = standardize(X_test, X_mu, X_std)
    
    # Find min and max of neural network output, which
    # is the output of the drift coefficient (i.e., Y[:,0:nx])
    Y_mu, Y_std = find_mu_std(Y_train)
    
    # Normalize Y[:,0:nx]
    Y_train_scaled = standardize(Y_train, Y_mu, Y_std)
    Y_val_scaled = standardize(Y_val, Y_mu, Y_std)
    Y_test_scaled = standardize(Y_test, Y_mu, Y_std)
    
    # Ensure float32 data type and save
    X_train_scaled = np.float32(X_train_scaled)
    X_val_scaled = np.float32(X_val_scaled)
    X_test_scaled = np.float32(X_test_scaled)
    Y_train_scaled = np.float32(Y_train_scaled)
    Y_val_scaled = np.float32(Y_val_scaled)
    Y_test_scaled = np.float32(Y_test_scaled)
    X_mu = np.float32(X_mu)
    X_std = np.float32(X_std)
    Y_mu = np.float32(Y_mu)
    Y_std = np.float32(Y_std)

    # Save data
    np.save(os.path.join(g2_path, 'X_train_g2.npy'), X_train_scaled)
    np.save(os.path.join(g2_path, 'X_val_g2.npy'), X_val_scaled)
    np.save(os.path.join(g2_path, 'X_test_g2.npy'), X_test_scaled)
    np.save(os.path.join(g2_path, 'Y_train_g2.npy'), Y_train_scaled)
    np.save(os.path.join(g2_path, 'Y_val_g2.npy'), Y_val_scaled)
    np.save(os.path.join(g2_path, 'Y_test_g2.npy'), Y_test_scaled)
    np.save(os.path.join(g2_path, 'X_mu_g2.npy'), X_mu)
    np.save(os.path.join(g2_path, 'X_std_g2.npy'), X_std)
    np.save(os.path.join(g2_path, 'Y_mu_g2.npy'), Y_mu)
    np.save(os.path.join(g2_path, 'Y_std_g2.npy'), Y_std)

    return [X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, 
            Y_val_scaled, Y_test_scaled, X_mu, X_std, Y_mu, Y_std]
############################################################################### 
    
###############################################################################
# Loss function for estimating drift coefficient for testing
def g2_test_loss(NN, y_true, x_test):
    
    # Get predictions of neural network
    y_pred = NN(x_test)
    
    # Calculate squared error between true output of drift coefficient
    # and UT/neural network estimated output
    squared_difference = tf.square(y_true - y_pred)
    
    # Get mean of squared error
    return np.mean(tf.reduce_mean(squared_difference, axis=-1).numpy())
###############################################################################

############################################################################### 
# Function that trains multiple neural networks and saves relevant outputs
def train_multiple_NNs_g2(X_train, X_val, X_test, Y_train, Y_val, Y_test,
                          n_hidden_layers, n_hidden_nodes, nx, nu, g2_path):
    
    # Get input and output dimensions of neural network
    input_dim = nx + nu
    output_dim = nx

    # Train neural networks that represent g1
    for nhl in n_hidden_layers:
        for nhn in n_hidden_nodes:
            
            # Create path where model will be saved
            path = os.path.join(g2_path, str(nhl) + "_HL_" + str(nhn) + "_Nodes/")
            os.mkdir(path)
            
            # Create neural network
            NN = create_NN(input_dim, output_dim, nhn, nhl)
            
            # Compile neural network
            NN = compile_NN(NN, "mean_squared_error")
            
            # Train neural network
            start = time.time() # Record start time
            NN = train_single_NN(NN, X_train, Y_train, X_val, Y_val)
            end = time.time() # Record end time
            total_time = end - start
            
            # Save neural network
            save_NN(NN, path)
            
            # Record losses
            train_loss = np.asarray(NN.history.history['loss'])
            val_loss = np.asarray(NN.history.history['val_loss'])
            test_loss = g2_test_loss(NN, Y_test, X_test)
            
            # Save loss
            np.save(path + "Train_Loss.npy", train_loss)
            np.save(path + "Val_Loss.npy", val_loss)
            np.save(path + "Test_Loss.npy", test_loss)
            
            # Save total run time
            np.save(path + "total_time.npy", total_time)

###############################################################################
            
###############################################################################
# Hidden physics neural network function for colloidal self-assembly system.
def CSA_NN(NN, states, xu_mu, xu_std, g_mu, g_std):
    
    # Pre-allocate
    states_scaled = np.zeros((1,2))
    
    # Scale states
    states_scaled[0,0] = (states[0]-xu_mu[0])/(xu_std[0])
    states_scaled[0,1] = (states[1]-xu_mu[1])/(xu_std[1])
    
    # Get NN output
    output = NN(states_scaled).numpy()[0][0]
    
    # Scale output and return value
    return (output)*(g_std[0])+g_mu[0]         
            
# Function that plots reconstructed hidden physics for colloidal self-assembly
# system       
def plot_reconstruction_CSA(NN, xu_mu, xu_std, g_mu, g_std, hp_type, path):
    
    # Minimum and maximum state values
    x_min = 0
    x_max = 5.1
    
    # Choose some number of state values for plotting. We choose 1000 in the
    # paper
    x = np.linspace(x_min, x_max, 1000)
    
    # Minimum and maximum exogenous input values
    u_min = 0.5
    u_max = 4.0
    
    # Choose some number of inputs for plotting. We choose 8 in the paper
    u = np.linspace(u_min, u_max, 8)
    
    # Create plots
    for j in range(0, len(u)):
        
        # Initialize
        g_true_list = []
        g_pred_list = []
        
        for i in range(0, len(x)):
            
            # Get state
            states= np.asarray([x[i], u[j], 0])
            
            # Get true prediction
            _, g1_true, g2_true = stoch_dyn_CSA(states)
            
            if hp_type == "g1":
                g_true_list.append(g1_true)
            elif hp_type == "g2":
                g_true_list.append(g2_true)
      
            # Get NN prediction
            g_pred_list.append(CSA_NN(NN, states, xu_mu, xu_std, g_mu, g_std))
        
        # Plot
        plt.figure(1)
        if j == 0:
            plt.plot(x, g_pred_list, color="red", linewidth=2, label="SPINN")
            plt.plot(x, g_true_list, color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x, g_pred_list, color="red", linewidth=2, label=None)
            plt.plot(x, g_true_list, color = "black", linestyle=":", linewidth=3, label=None)
        plt.legend(fontsize=16)
        plt.xlim([x_min, x_max])
        plt.xlabel("$x$", fontsize=20)
        if hp_type == "g1":
            plt.ylabel("$g_1(x,u)$", fontsize=20)
        elif hp_type == "g2":
            plt.ylabel("$g_2(x,u)$", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path, hp_type + "_CSA.png"))
###############################################################################

###############################################################################
# Hidden physics neural network function for Lotka-Volterra system.
def LVE_NN(NN, states, x_mu, x_std, g_mu, g_std):
    
    # Pre-allocate
    states_scaled = np.zeros((1,2))
    
    # Scale states
    states_scaled[0,0] = (states[0]-x_mu[0])/(x_std[0])
    states_scaled[0,1] = (states[1]-x_mu[1])/(x_std[1])
    
    # Get NN output
    output_1 = NN(states_scaled).numpy()[0][0]
    output_2 = NN(states_scaled).numpy()[0][1]
    
    # Scale output and return value
    return (output_1)*(g_std[0])+g_mu[0], (output_2)*(g_std[1])+g_mu[1]              
        
        
# Function that plots reconstructed hidden physics for colloidal self-assembly
# system       
def plot_reconstruction_LVE(NN, x_mu, x_std, g_mu, g_std, hp_type, path):
    
    # Minimum and maximum state values
    x_min = [0,0]
    x_max = [2,2]
    
    # Choose to "plot" one dimension and "hold" another dimension. The choices
    # here are arbitrary. Really this type of plot should be done in MATLAB
    # or some tool that has better 3D visualization options than python.
    x1 = np.linspace(x_min[0], x_max[0], 1000) # "Plotting"
    x2 = np.linspace(x_min[1], x_max[1], 8) # "Holding"
    
    # Create plots
    for j in range(0, len(x2)):
        
        # Initialize
        g_true_list_1 = []
        g_true_list_2 = []
        
        g_pred_list_1 = []
        g_pred_list_2 = []
        
        for i in range(0, len(x1)):
            
            # Get state
            states= np.asarray([x1[i], x2[j], 0, 0])
            
            # Get true prediction
            _, g1_true, g2_true = stoch_dyn_LVE(states)
            
            if hp_type == "g1":
                g_true_list_1.append(g1_true[0])
                g_true_list_2.append(g1_true[1])
            elif hp_type == "g2":
                g_true_list_1.append(g2_true[0])
                g_true_list_2.append(g2_true[1])
      
            # Get NN prediction
            pred = LVE_NN(NN, states, x_mu, x_std, g_mu, g_std)
            g_pred_list_1.append(pred[0])
            g_pred_list_2.append(pred[1])
        
        # Plot
        plt.figure(1)
        if j == 0:
            plt.plot(x1, g_pred_list_1, color="red", linewidth=2, label="SPINN")
            plt.plot(x1, g_true_list_1, color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x1, g_pred_list_1, color="red", linewidth=2, label=None)
            plt.plot(x1, g_true_list_1, color = "black", linestyle=":", linewidth=3, label=None)
        plt.legend(fontsize=16)
        plt.xlim([x_min[0], x_max[0]])
        plt.xlabel("$x_1$", fontsize=20)
        if hp_type == "g1":
            plt.ylabel("$g_1(x_1,x_2)_1$", fontsize=20)
        elif hp_type == "g2":
            plt.ylabel("$g_2(x_1,x_2)_1$", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path, hp_type + "_1_LVE.png"))
        
        plt.figure(2)
        if j == 0:
            plt.plot(x1, g_pred_list_2, color="red", linewidth=2, label="SPINN")
            plt.plot(x1, g_true_list_2, color = "black", linestyle=":", linewidth=3, label="True")
        else:
            plt.plot(x1, g_pred_list_2, color="red", linewidth=2, label=None)
            plt.plot(x1, g_true_list_2, color = "black", linestyle=":", linewidth=3, label=None)
        plt.legend(fontsize=16)
        plt.xlim([x_min[0], x_max[0]])
        plt.xlabel("$x_1$", fontsize=20)
        if hp_type == "g1":
            plt.ylabel("$g_1(x_1,x_2)_2$", fontsize=20)
        elif hp_type == "g2":
            plt.ylabel("$g_2(x_1,x_2)_2$", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path, hp_type + "_2_LVE.png"))        
###############################################################################        