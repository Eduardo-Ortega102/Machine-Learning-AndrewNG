function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


%{
	Este es el enfoque iterativo de la normalización

for i=1:size(X, 2)
	mu(1,i) = mean(X(:,i));
	sigma(1,i) = std(X(:,i));
	X_norm(:,i) = (X(:,i) - mu(1,i)) ./ sigma(1,i);
end

%}

%{
	Este es el enfoque vectorizado de la normalización

#1. calculamos el vector mu (en este caso, es un vector de 1x2 (1 fila, 2 columnas))
#2. calculamos el vector sigma (en este caso, es un vector de 1x2 (1 fila, 2 columnas))

X es de dimension 47x2, hay que crear vectores mayores para que se pueda operar.
#3. crear 'mu_matrix' y 'sigma_matrix', que serán vectores de 47x2 (la misma dimension que X)

%}

mu = mean(X);	
sigma = std(X); 
m = size(X, 1); % obtener la primera dimension de la matriz (numero de filas)
mu_matrix = ones(m, 1) * mu; 		% 47x1 * 1x2 = 47x2
sigma_matrix = ones(m, 1) * sigma;	% 47x1 * 1x2 = 47x2
X_norm = (X - mu_matrix) ./ sigma_matrix;


% ============================================================

end
