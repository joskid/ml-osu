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

% Maximum number of epochs
iters = 1000;

%compute theta using gradient descent
[theta,batchCostVals] = batchGradientDescent(X, y, theta, alpha,iters);

%Display theta found by gradient descent
fprintf('Theta values as computed by gradient descent are %s \n',theta);

testData = load('regression-test.csv');
testX = testData(:,1:end-1);
testY = testData(:,end);
testX = [ones(length(testY),1),testX];
squaredError = computeCostBatch(testX,testY,theta);
fprintf('The sum of squared error on test data is %d',squaredError);

% Checking for convergence, cost VS iterations plot
figure;
hold on;
plot(1:length(batchCostVals),batchCostVals,'-b', 'LineWidth', 2);
fprintf('length of batch %d',length(batchCostVals));
xlabel('Number of Epochs');
ylabel('SEE Value');



