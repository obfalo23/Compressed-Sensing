clear all
close all
clc
load("cs.mat")

% Show the true data vector
figure
plot(x);
title("true data vector")
disp("True data vector l1-norm:")
disp(norm(x,1))

% Parameters
N = n; % Data size
step_size = 1;
epsilon = 1e-15; % Stop criterion
gamma = 0.1;
step_size_exp_parameter = 50;
K = 5000; % Max steps of simulation

% Use a twice as tall expanded matrix to be able to differentiate an
% otherwise complex matrix
F_us_exp = [real(F_us);imag(F_us)];
X_us_exp = [real(X_us);imag(X_us)];

% Create the initial x_est
x_est = zeros(128,K);

% Calculate initial error
error = zeros(K,1);
error(1) = norm(F_us*x_est(:,1) - X_us, 2);
disp("Initial error:")
disp(error(1))
best_error = error(1); % tracking the best error

% Descent till stop criterion is met on l2 error
k = 1;
tStart = cputime;
while norm(F_us*x_est(:,k) - X_us, 2) > epsilon && k < K
    % Calculate first derivatives (direction)
    nabula = (2*(F_us_exp'*F_us_exp)*x_est(:,k) - 2*F_us_exp'*X_us_exp) / norm(F_us*x_est(:,k) - X_us,2)  + gamma * sign(x_est(:,k));
    
    % Calculate step size with decreasing exponential
    step_size = exp(-(k)/step_size_exp_parameter)/(k+1);
    
    % Subgradient descent and check element wise for non-negativity constraint 
    % and reduce nabula iteratively to get into solution space
    for i=1:1:128
        j = 1;
        while j < 100
            x_est(i,k+1) = x_est(i,k) - step_size*nabula(i,1);
            if x_est(i,k+1) < 0
                nabula(i,1) = 0.01/j*step_size*nabula(i,1);
            else 
                break;
            end
            j = j + 1;
        end
    end
    
    % Calculate error and decide on best error
    error(k) = norm(F_us*x_est(:,k+1) - X_us, 2);
    if error(end) <= best_error
        best_x_est = x_est(:,k);
        best_error = error(k);
    end

    k = k + 1;
    % Progress update, comment out if not needed
    if mod(k,1000) == 0
        k
        -log(error(k-1))
    end
end
tEnd = cputime - tStart;
disp("CPU time since start of loop")
disp(tEnd);

disp("Steps to get to stopping criterion:")
disp(k)

disp("Final error, l2-norm of cost function:") 
disp(norm(F_us*best_x_est - X_us, 2))

disp("Error with true vector")
disp(norm(best_x_est-x,2))

figure;
plot(error)
yscale("log")
title("Error")

figure;
plot(best_x_est)
title("Estimated x using gradient descent for constrained problem")
