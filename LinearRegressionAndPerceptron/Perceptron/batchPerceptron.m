% Batch Perceptron algorithm
function [weights,errorMatrix] = batchPerceptron(X,y,weights,lambda,epoches)
m = size(X,1);
no_of_features = size(X,2);

errorMatrix = zeros(epoches,1);
 
j = 0;
    while true
        j = j + 1;
        errors = 0;
        delta = zeros(no_of_features,1);
        for i = 1:m
            temp = X(i,:) * weights;
            if temp * y(i) <= 0
                errors = errors + 1;
                delta = delta - (y(i) * X(i,:)');
            end
        end
        delta = delta ./ m;
        weights = weights - (lambda * delta);
        errorMatrix(j) = errors;
        if (norm(delta) < 0.00000000001)
            fprintf('Number of epoches are %d',j);
            return;
        end
        
    end

end