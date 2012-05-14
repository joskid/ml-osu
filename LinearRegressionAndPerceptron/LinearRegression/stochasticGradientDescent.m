function [theta,costVals] = stochasticGradientDescent(X,y,theta,alpha,iters)
m = length(y);
counter = 0;
costVals = zeros(iters,1);
    for i = 1:iters
        j = randperm(size(X,1));
        tempCounter = 0;
        for temp = j    
            tempCounter = tempCounter + 1;
            delta = ((X(temp,:)') * ((X(temp,:) * theta) - y(temp)));
            theta = theta - (alpha * delta);
            %saving the cost values for each of theta
            if tempCounter == 4500
                counter = counter + 1;
                costVals(counter) = computeCostStochastic(X,y,theta);
            end
        end
        
    end
end