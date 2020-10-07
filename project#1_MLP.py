from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # np.random.randn(shape)
        # - Return a sample (or samples) from the “standard normal” distribution following shape
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)  # (4,10)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']  # 위에서 지정한 random값 들어가있음.
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape  # 각각 행 개수, 열 개수(5,4)

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        Z1 = np.dot(X, W1) + b1  # Z1, 차후 Derivative에 재사용, 5 X 10 matrix
        Index = 0
        # activation function Relu, np.maximum 사용 없이 구현
        for i in Z1:
            for j in i:
                if j < 0:
                    i[Index] = 0.0
                    Index += 1
                else:
                    Index += 1
                    continue
            Index = 0
        scores = np.dot(Z1, W2) + b2  # scores는 Z2에 해당한다.
        # print(scores)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        """y에 exponential 취한 후 normalization해서 각각 확률을 갖는 벡터로 탄생"""

        ExpScores = np.exp(scores)  # scores에 exponential 취함
        SoftMaxValue = ExpScores / np.sum(ExpScores, axis=1, keepdims=True)  # softmax matrix 생성, 확률 값 가짐 (5x3)
        LogSMValue = np.zeros((N,))  # SoftMaxValue에서 y[i]번째(각 input의 정답 index) 값을 가져간다. 그 후 5x1 matrix 구성
        for i in range(N):
            LogSMValue[i] = np.log(SoftMaxValue[i, int(y[i])])
        data_loss = -(np.sum(LogSMValue) / N)  # L(W) 구해준다.
        L2RLoss = reg * (np.sum(W2 * W2) + np.sum(W1 * W1))  # L-2 Regularization 생성
        loss = data_loss + L2RLoss  # 우리가 찾는 최종 total loss
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for i in range(N):  # p-y
            SoftMaxValue[i, y[i]] -= 1
        dLdZ2 = SoftMaxValue / N
        TransZ1 = Z1.T
        dLdW2 = np.dot(TransZ1, dLdZ2)
        dLdb2 = dLdZ2.sum(axis=0)  # Cx1 Matrix로 만들어줌
        dZ2dp1 = W2.T
        dTemp = np.dot(dLdZ2, dZ2dp1)
        N, H = Z1.shape  # 각각 행 개수, 열 개수(5,10)
        for i in range(N):
            for j in range(H):
                if (Z1[i, j] == 0):
                    dTemp[i, j] = 0
        dLdW1 = np.dot(X.T, dTemp)  # dLdZ1 * dZ1dW1
        dLdb1 = np.sum(dTemp, axis=0, keepdims=True)

        # Add the gradient of the regularization
        grads['W1'] = dLdW1 + (reg * W1)
        grads['W2'] = dLdW2 + (reg * W2)
        grads['b2'] = dLdb2
        grads['b1'] = dLdb1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]  # Training data input 수
        iterations_per_epoch = max(num_train / batch_size, 1)  # of Epoch

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            # - See [ np.random.choice ]											  #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            RandomMinBatch = np.random.choice(num_train, batch_size)  # toy data 기준 0~4중 임의의 숫자 200개 추출
            X_batch = X[RandomMinBatch]
            y_batch = y[RandomMinBatch]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y_batch)  # loss function you completed above
            loss_history.append(loss)  # loss값 loss_history에 추가

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Weight과 bias를 update한다.
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            Tempb1 = learning_rate * grads['b1']
            Tempb2 = learning_rate * grads['b2']
            self.params['b1'] -= Tempb1.ravel()
            self.params['b2'] -= Tempb2.ravel()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # print loss value per 100 epoch
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        # perform forward pass and return index of maximum scores				  #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ForwardPass = self.loss(X)
        y_pred = np.argmax(ForwardPass, axis=1)  # 가장 큰 값의 index를 y_pred에 저장한다.

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred


