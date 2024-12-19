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

% Show X_us
% figure;
% plot(real(X_us));
% title("X us / F_us*x");

% Create the initial x_est
x_est = zeros(128,1000);

error = double.empty;


% Parameters
step_size = 1;
epsilon = 0.001; % Stop criterion
gamma = 0.1;
max_steps = 50;
best_error = 1000; % Make it high so it always chooses the minimum.

F_us_exp = [real(F_us);imag(F_us)];
X_us_exp = [real(X_us);imag(X_us)];

k = 1;
disp("Initial error:")
% Calculate error
error = [error; norm(F_us*x_est(:,1) - X_us, 2)];
disp(error)

tStart = cputime;
while norm(F_us*x_est(:,k) - X_us, 2) > epsilon && k < max_steps
    % Calculate first derivatives (direction)
    nabula = (2*F_us_exp'*F_us_exp*x_est(:,k) - 2*F_us_exp'*X_us_exp) / norm(F_us*x_est(:,k) - X_us,2)  + gamma * sign(x_est(:,k));
    
    % Calculate step size
    step_size = 1/(k+1);

    % Calculate new x
    for i=1:1:128
        j = 3;
        while j < 100
            x_est(i,k+1) = x_est(i,k) - step_size*nabula(i,1);
            if x_est(i,k+1) < 0
                %nabula(i,1) = 1/j*step_size*nabula(i,1);
                nabula(i,1) = 0;
            else 
                break;
            end
            j = j + 1;
        end
    end
    
    % Calculate new x
    x_est(:,k+1) = x_est(:,k) - step_size*nabula;
    
    % Calculate error
    error = [error; norm(F_us*x_est(:,k+1) - X_us, 2)];
    
    if error(end) <= best_error
        best_x_est = x_est(:,k);
        best_error = error(end);
    end

    k = k + 1;
end
tEnd = cputime - tStart;
disp("CPU time since start of loop")
disp(tEnd);

disp("Steps to get to stopping criterion:")
disp(k)

disp("Final error, l2-norm of cost function:") 
disp(norm(F_us*best_x_est - X_us, 2))

disp("Final l1-norm of x:") 
disp(norm(best_x_est, 1))

disp("Error with true vector")
disp(norm(best_x_est-x,2))

figure;
plot(error)
yscale("log")
title("Error")

figure;
plot(best_x_est)
title("Estimated x using gradient descent for constrained problem")
