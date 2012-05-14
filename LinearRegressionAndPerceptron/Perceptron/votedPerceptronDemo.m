% loading the data
data = load('iris-twoclass.csv');

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

% Compute weights using batch perceptron algorithm
epoches = 100;
[weights,errorMatrix,counts] = votedPerceptron(X,y,epoches);
[maxval,maxind] = max(counts);

% plot decision boundary
Z= [weights(2,maxind),weights(3,maxind)];
B = weights(1,maxind); 
plotpc(Z,B);
title('using best of the weight in voted perceptron algorithm');
hold off;

%plot errors
figure;
plot(1:length(errorMatrix),errorMatrix,'-b', 'LineWidth', 2);
fprintf('length of batch %d',length(errorMatrix));
xlabel('Number of Epochs');
ylabel('Misclassified points');

%plotting the data againf for averaged weight
y = data(:,1);
X = data(:,2:end);
positives = find(y == 1);
negatives = find(y == -1);
figure;
hold on;
plot(X(positives,1),X(positives,2),'b*','LineWidth',2,'MarkerSize',5);
plot(X(negatives,1),X(negatives,2),'ro','LineWidth',2,'MarkerSize',5);

weightedAvg = counts' * weights';

% plot decision boundary
Z= [weightedAvg(1,2),weightedAvg(1,3)];
B = weightedAvg(1,1); 
plotpc(Z,B);
title('using average weight in voted perceptron algorithm');
hold off;