function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % number of training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%
%   PART 1
%
%{
			Some dimentions

		X				= 5000x400
		y 				= 5000x1  (labels vector)
        Theta1          = 25x401 number of neurons x (input + bias)
        Theta2          = 10x26  number of neurons x (hidden layer result + bias)
%}

y_matrix = eye(num_labels)(y,:) % producing a matrix of 5000x10

a1 = [ones(m, 1), X]; % 5000x401 (add bias)
z2 = a1 * Theta1'; % 5000x401 * 401x25 = 5000x25
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2]; % 5000x26 (add bias)
z3 = a2 * Theta2'; % 5000x26 * 26x10 = 5000x10
a3 = sigmoid(z3);

%{
			Some dimentions
y_matrix        = 5000x10
a3(hypotesis)   = 5000x10
J               = 1x1 (scalar)
%}

%{
    
    METHOD 1: double-sum of element-wise product
%}
left_term = -y_matrix .* log(a3); % 5000x10 .* 5000x10 = 5000x10
right_term = (1 - y_matrix) .* log(1 - a3); %  5000x10 .* 5000x10 = 5000x10
double_sum = sum(sum(left_term - right_term)); % scalar
J = (1/m) * double_sum; % scalar


%{
    METHOD 2: matrix product with trace (sum of the main diagonal elements)

y_matrix = y_matrix'; % 10x5000
left_term = -y_matrix * log(a3); % 10x5000 * 5000x10 = 10x10
right_term = (1 - y_matrix) * log(1 - a3); %  10x5000 * 5000x10 = 10x10
J = (1/m) * trace(left_term - right_term)
%}

regularization_term = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + regularization_term;






%
%   PART 2
%
%{
			Some dimentions
y_matrix        = 5000x10
a3(hypotesis)   = 5000x10
a2              = 5000x26
a1              = 5000x401
Theta1          = 25x401 number of neurons x (input + bias)
Theta2          = 10x26  number of neurons x (hidden layer result + bias)
%}

d3 = a3 - y_matrix; % 5000x10
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); % 5000x10 * 10x25 .* 5000x25

Delta1 = d2' * a1; % 25x5000 * 5000x401 = 25x401
Delta2 = d3' * a2; % 10x5000 * 5000x26  = 10x26

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;





%
%   PART 3
%
Theta1_grad_regularization = (lambda/m) * Theta1(:, 2:end);
Theta2_grad_regularization = (lambda/m) * Theta2(:, 2:end);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1_grad_regularization;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2_grad_regularization;

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
