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
theta = zeros(n+1,1);

% Setting alpha and the number of iterations
alpha = 0.01;
iters = 50;

%compute initial cost
cost = computeCostStochastic(X,y,theta);

%compute theta using gradient descent
[theta,costVals] = stochasticGradientDescent(X, y, theta, alpha,iters);

%Display theta found by gradient descent
fprintf('Theta values as computed by gradient descent are %s \n',theta);

testData = load('regression-test.csv');
testX = testData(:,1:end-1);
testY = testData(:,end);
testX = [ones(length(testY),1),testX];
squaredError = computeCostStochastic(testX,testY,theta);
fprintf('The sum of squared error on test data is %d',squaredError);

% Checking for convergence, cost VS iterations plot
figure;
plot(1:numel(costVals),costVals,'-b', 'LineWidth', 2);
xlabel('Number of epochs');
ylabel('SEE Value');




