function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVERANDOM Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVERANDOM(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

m = size(X, 1);   % Number of training examples
r = size(Xval,1)  % the number of validation examples


% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


maxIter = 50;
for i = 1:m 
    partial_error_train = zeros(1, maxIter);
    partial_error_val   = zeros(1, maxIter);
    for j = 1:maxIter
        random_train = randperm(m, i); % get a row vector containing 'i' unique integers selected randomly from 1 to m (inclusive)
        random_val   = randperm(r, i); % get a row vector containing 'i' unique integers selected randomly from 1 to r (inclusive)
        X_train = X(random_train, :); % select rows using 'random_train' as index
        y_train = y(random_train);    % select rows using 'random_train' as index
        X_val = Xval(random_val, :);     % select rows using 'random_val' as index
        y_val = yval(random_val);     % select rows using 'random_val' as index
        theta = trainLinearReg(X_train, y_train, lambda);
        partial_error_train(j) = linearRegCostFunction(X_train, y_train, theta, 0);
        partial_error_val(j)   = linearRegCostFunction(X_val, y_val, theta, 0);
    end
    error_train(i) = mean(partial_error_train);
    error_val(i) =  mean(partial_error_val);
end

% -------------------------------------------------------------
% =========================================================================

end
