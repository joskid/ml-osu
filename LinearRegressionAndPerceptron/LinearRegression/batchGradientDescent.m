function [theta,batchCostVals] = batchGradientDescent(X,y,theta,alpha,iters)
m = length(y);
batchCostVals = zeros(iters,1);

    for i = 1:iters
        
        delta = (1/m) * ((X') * ((X * theta) - y));
        theta = theta - (alpha * delta);
        
        %saving the cost values for each of theta
        batchCostVals(i) = computeCostBatch(X,y,theta);
     
    end
end