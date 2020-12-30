function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% Need to return the following variables correctly. Set J to the cost
J = 0;

for i = 1:m
    %disp(i);
    J = J + ((theta(1) + theta(2) * X(i, 2)) - y(i, 1)) .^ 2;
    %disp(y(i, 1));
end
J = J / (2 * m);

% =========================================================================

end
