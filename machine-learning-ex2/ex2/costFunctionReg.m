function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%{
							Some dimentions

					X 			=  118x28 matrix
					X' 			=  28x118 matrix
					theta		=	 28x1 vector
					X * theta	=	118x1 vector 
					y 			=   118x1 vector
					y' 			=   1x118 vector
					grad 		=    28x1 vector
%}

n = size(theta, 1); % number of features

hypotesis = sigmoid(X * theta);
J_regularization_term = (lambda/(2*m)) * sum(theta(2:n) .^ 2);
J = (1/m) * (-y' * log(hypotesis) - (1 - y)' * log(1 - hypotesis)) + J_regularization_term;


grad_regularization_term = (lambda/m) .* theta;
grad_general_term = (1/m) * (X' * (hypotesis - y));
grad(1) = grad_general_term(1);
grad(2:n) = grad_general_term(2:n) + grad_regularization_term(2:n);

% =============================================================

end
