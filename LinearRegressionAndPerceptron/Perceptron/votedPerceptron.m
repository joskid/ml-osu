% Voted Perceptron algorithm
function [weights,errorMatrix,counts] = votedPerceptron(X,y,epoches)
m = size(X,1);
no_of_features = size(X,2);
counts = zeros(m,1);
weights = zeros(no_of_features,m);
errorMatrix = zeros(epoches,1);
n = 1;

    for i = 1:epoches
        errors = 0;
        rand = randperm(m);
        for j = rand
            
            yhat = X(j,:) * weights(:,n);
            if yhat * y(j) <= 0
                errors = errors + 1;
                weights(:,n+1) = weights(:,n) + (y(j) * X(j,:)');
                counts(n+1) = 1; 
                n = n + 1;
            else
                counts(n) = counts(n) + 1;
            end
            
        end
        errorMatrix(i) = errors;
    end

end