% Loading data
data = load('regression-train.csv');

%Assignment of feature and target vectors
X = data(:,1:end-1);
y = data(:,end);

% 'm' is size of the dataset
m = size(X,1);

% 'n' is the number of features
n = size(X,2);

%Adding vector of 1's to X and initialization of theta
X = [ones(m,1),X];
theta = ones(n+1,1);

% Setting alpha and the number of iterations
alpha = 0.03;

% Maximum number of epochs
iters = 50;

%compute theta using gradient descent
[batchTheta,batchCostVals] = batchGradientDescent(X, y, theta, alpha,iters);

%compute theta using gradient descent
[stochasticTheta,stochasticCostVals] = stochasticGradientDescent(X, y, theta, alpha,iters);

figure;
hold on;
% plotting batch GD SEE values
plot(1:length(batchCostVals),batchCostVals,'-b', 'LineWidth', 2);
xlabel('Number of Epochs');
ylabel('SEE Value');

% plotting stochastic GD SEE values
tempVals = zeros(iters,1);
%j = 1;
%for i = 1:m-1:length(stochasticCostVals)
 %   tempVals(j) = stochasticCostVals(j);
  %  j = j + 1;
%end
plot(1:numel(stochasticCostVals),stochasticCostVals,'-r', 'LineWidth', 2);
legend('Batch GD','Stochastic GD');
hold off;























