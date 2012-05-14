% loading the data
data = load('twogaussian.csv');

% plotting the data
y = data(:,1);
X = data(:,2:end);
positives = find(y == 1);
negatives = find(y == -1);
figure;
hold on;
plot(X(positives,1),X(positives,2),'b*','LineWidth',2,'MarkerSize',5);
plot(X(negatives,1),X(negatives,2),'ro','LineWidth',2,'MarkerSize',5);

% Dimensions
m = length(y);
n = size(X,2);

% Adding ones to the feature vector
X = [ones(m,1),X];
initialWeights = zeros(n+1,1);

% Compute weights using batch perceptron algorithm
lambda = 1;
epoches = 35;
[weights,errorMatrix] = batchPerceptron(X,y,initialWeights,lambda,epoches);

% plot decision boundary
Z= [weights(2,1),weights(3,1)];
B = weights(1); 
plotpc(Z,B);
hold off;

%plot errors
figure;
plot(1:length(errorMatrix),errorMatrix,'-b', 'LineWidth', 2);
fprintf('length of batch %d',length(errorMatrix));
xlabel('Number of Epochs');
ylabel('Misclassified points');