function [ cost ] = computeCostStochastic( X, y, theta )

cost = (1/(2)) * (sum(((X * theta) - y) .^2));

end

