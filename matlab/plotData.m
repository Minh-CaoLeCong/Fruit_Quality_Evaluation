function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   number of pixels and weight.

figure; % open a new figure window

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Weight of fruit');
xlabel('Number of pixels');



end