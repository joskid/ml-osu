function [ cost ] = computeCostBatch( X, y, theta )

m = size(X,1);

cost = (1/(2*m)) * (sum(((X * theta) - y) .^2));

end

