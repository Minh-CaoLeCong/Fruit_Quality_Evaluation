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
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

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
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Plot the convergence graph
figure;
fprintf('Plot the convergence graph ...\n')
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
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
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
