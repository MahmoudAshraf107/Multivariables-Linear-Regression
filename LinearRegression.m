%% Adjustment the Work Environment
clc
clear 
close all
%% load data and set the parameters
Data = load ('data2.txt');                   % load the dataset
alpha=0.1;                                   % Learning rate coefficient 
iterations=100;                              % Number of training iterations
%% Main code
data=((Data-mean(Data))./std(Data));                  % Data rescaling
[m,n]=size(data);                                     % get the size of the dataset
NewData=[ones(m,1)  data];                            % add new column to the matrix (X0)
x=NewData(:,1:n);                                     % Export the input matrix form the dataset
y=NewData(:,n+1);                                     % Export the output matrix form the dataset
theta=zeros(1,n);                                     % Assume Theta vector with zeros  
cost0=J(x,y,theta);                                   % Compute the cost with the assuemed theta
[finaltheta,cost]=GD(x,y,theta,alpha,iterations,n);   % Call the Gradient Descent function 
finalcost=cost(1,iterations);                         % Call the cost function 
%% find the best fit line 
for i=n:-1:2                                                 % for loop for multi feature examples
A(:,i)=linspace (min(-NewData(:,n)),max(NewData(:,n)),100);  % take 100 values with equal intervals from the input1 bedrooms=NewData(:,n)
B(:,i)=finaltheta(1,1)+(finaltheta(1,2)*A(:,i));             % find the expected vaules of the output at X
end 
%% Plotting the results 
for i=n:-1:2                                     % for loop for multi feature examples
name=['Graph ',num2str(i-1)];
figure ("Name",name)                             % Create new figure 
hold on
grid on                                          % Create grid in the figure 
scatter(NewData(:,i),NewData(:,n+1),'filled');   % Scatter plot between input in x-axis and output in y-axis price=NewData(:,n+1)
xlabel(['parameter ',num2str(i-1)]);                         % x-axis label
ylabel({'Price'});                               % y-axis label
plot(A(:,i),B(:,i),'r','linewidth',2);           % Plot the best fit line with redline and width 2 on the same figure 
hold off
end 
figure ("Name",'Cost')                           % Create new figure
plot(1:iterations,cost,'r','linewidth',2);       % Plot the rerlation between number of iterations and cost 
grid on                                          % Create grid in the figure 
xlabel({'Iterations'});                          % x-axis label
ylabel({'Cost'});                                % y-axis label
grid on                                          % Create grid in the figure 
figure ("Name",'3D Graph')                       % Create new figure
plot3(A(:,n-1),A(:,n),B(:,n),'r','LineWidth',2); % 3d plot for the last two features with the result
xlabel(['parameter',num2str(n-2)]);              % x-axis label
ylabel(['parameter',num2str(n-1)]);              % y-axis label
zlabel({'price',});                              % z-axis label
box on                                           % Create Box in the figure 
grid on                                          % Create grid in the figure 
%% Ending the Editor
fprintf(' \t\t\t\t\t\t\t\t\t\t\t Machine Learning \n')
fprintf(' \t\t\t\t\t\t\t\t\t\t\t Linear Regression \n')
fprintf(' \t\t\t\t\t\t\t\t\t\t  Mahmoud Ashraf Mahmoud \n')
%% Functions
function J=J(input,output,theta)                                 % create function with output J called J and three inputs 
         J=sum(((input*theta')-output).^2)/(2*length(input));    % Function equation (cost equation) 
         return                                              
end                                                                
function [theta,cost]=GD(input,output,theta,alpha,iterations,n)            % Gradient Descent function with outputs (theta, cost) called GD with six inputs 
         temp=zeros(size(theta));                                          % create zeros matrix with thenat size 
         cost=zeros(1,iterations);                                         % create cost as zeros matrix with iterations size
         for i=1:1:iterations                                              % for loop from i=1 to interations 
             error=(input*theta')-output;                                  % compute the error 
             for c=1:n                                                     % for loop fron j=1 to n 
                 term=error.*input(:,c);                                   % Compute the term of h(x)-
                 temp(1,c)=theta(1,c)-((alpha/length(input))*sum(term));   % compute the new theta (Gardient Desent)
             end
         theta=temp;                                                       % get the new vales of theta 
         cost(1,i)=sum(((input*theta')-output).^2)/(2*length(input));      %compute the cost according to the new values of theta
         end 
         return
end
          