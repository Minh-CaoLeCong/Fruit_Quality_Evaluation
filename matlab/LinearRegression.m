% MATLAB 2020b

%% Linear Regression: Fruit Quality Evaluation

%  Instructions
%  ------------
%
%  This file contains:
%
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     featureNormalize.m
%     normalEqn.m
%
% x refers to the number of pixels of fruit
% y refers to the weight of fruit

%% Initialization
clear ; close all; clc
fprintf('Linear Regression: Fruit Quality Evaluation \n');

%% ============== Plotting and Feature Normalization ==============
data = load('Weight_AreaPixels_B_Immaturity_25_50.txt');
X = data(:, 1); 
y = data(:, 2);
m = length(y); % number of training examples

% Scale training examples
% X = X / 100000;
% y = y / 100;

% Scale features and set them to zero mean
fprintf('\nNormalizing Features ...\n');

% Mean normalization
[X mu sigma] = featureNormalize(X);
%[y mu sigma] = featureNormalize(y);

fprintf('Plotting Data ...\n')

% Plot Data
plotData(X, y);

%% =================== Cost and Gradient descent ===================

X = [ones(m, 1) X]; % Add intercept term to X
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 50;
alpha = 0.2;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

theta1 = zeros(2, 1);
theta2 = zeros(2, 1);
theta3 = zeros(2, 1);
theta4 = zeros(2, 1);
theta5 = zeros(2, 1);
theta6 = zeros(2, 1);
theta7 = zeros(2, 1);
theta8 = zeros(2, 1);

[theta1, J1] = gradientDescent(X, y, theta1, 0.001, iterations);
[theta2, J2] = gradientDescent(X, y, theta2, 0.003, iterations);
[theta3, J3] = gradientDescent(X, y, theta3, 0.01, iterations);
[theta4, J4] = gradientDescent(X, y, theta4, 0.03, iterations);
[theta5, J5] = gradientDescent(X, y, theta5, 0.1, iterations);
[theta6, J6] = gradientDescent(X, y, theta6, 0.3, iterations);
[theta7, J7] = gradientDescent(X, y, theta7, 1, iterations);
[theta8, J8] = gradientDescent(X, y, theta8, 3, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

% further testing of the cost function
J = computeCost(X, y, theta);
fprintf('With theta = %.6f\n', theta);
fprintf('Cost computed = %.6f\n', J);

% Plot the linear fit
hold on; % keep previous plot visible
fprintf('\nPlot the linear fit ...\n')
plot(X(:,2), X*theta, '-')

plot(X(:,2), X*theta1, '-', 'MarkerEdgeColor', 'b')
plot(X(:,2), X*theta2, '-', 'MarkerEdgeColor', 'g')
plot(X(:,2), X*theta3, '-', 'MarkerEdgeColor', 'r')
plot(X(:,2), X*theta4, '-', 'MarkerEdgeColor', 'c')
plot(X(:,2), X*theta5, '-', 'MarkerEdgeColor', 'm')
plot(X(:,2), X*theta6, '-', 'MarkerEdgeColor', 'y')
plot(X(:,2), X*theta7, '-', 'MarkerEdgeColor', 'k')
%plot(X(:,2), X*theta8, '-', 'MarkerEdgeColor', 'w')

legend('TData', '0.2', '0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1')
hold off % don't overlay any more plots on this figure

% Plot the convergence graph
figure;
fprintf('Plot the convergence graph ...\n')
plot(1:numel(J_history), J_history, 'LineWidth', 2);

hold on; % keep previous plot visible

plot(1:numel(J1), J1, '-b', 'LineWidth', 2);
plot(1:numel(J2), J2, '-g', 'LineWidth', 2);
plot(1:numel(J3), J3, '-r', 'LineWidth', 2);
plot(1:numel(J4), J4, '-c', 'LineWidth', 2);
plot(1:numel(J5), J5, '-m', 'LineWidth', 2);
plot(1:numel(J6), J6, '-y', 'LineWidth', 2);
plot(1:numel(J7), J7, '-k', 'LineWidth', 2);
%plot(1:numel(J8), J8, '-w', 'LineWidth', 2);

legend('0.2', '0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1')

xlabel('Number of iterations');
ylabel('Cost J');

%% ============= Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(530, 550, 100);
theta1_vals = linspace(90, 110, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(1, 3, 100))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);

plot(theta1(1), theta1(2), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta2(1), theta2(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta3(1), theta3(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta4(1), theta4(2), 'cx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta5(1), theta5(2), 'mx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta6(1), theta6(2), 'yx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta7(1), theta7(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

legend('J','0.2', '0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1')

%% ================ Normal Equations ================
data = load('Weight_AreaPixels_B_Immaturity_25_50.txt');
X = data(:, 1); 
y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1) X]; % Add intercept term to X

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('\n');
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);

% further testing of the cost function
J = computeCost(X, y, theta);
fprintf('With theta = %.6f\n', theta);
fprintf('Cost computed = %.6f\n', J);
fprintf('\n');