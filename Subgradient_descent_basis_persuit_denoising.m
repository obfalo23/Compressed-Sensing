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
epsilon = 0.01; % Stop criterion
gamma = 0.111;
max_steps = 50;
best_error = 1000; % Make it high so it always chooses the minimum.

F_us_exp = [real(F_us);imag(F_us)];
X_us_exp = [real(X_us);imag(X_us)];

disp("Initial error:")
disp(norm(F_us*x_est(:,1) - X_us, 2))

k = 1;
tStart = cputime;
while norm(F_us*x_est(:,k) - X_us, 2) + gamma*norm(x_est,1) > epsilon && k < max_steps
    % Calculate step size
    step_size = 1/k;

    % Calculate first derivatives (direction)
    %nabula = (2*F_us'*F_us*x_est(:,k) - 2*F_us'*X_us) / norm(F_us*x_est(:,k) - X_us);
    nabula = (2*F_us_exp'*F_us_exp*x_est(:,k) - 2*F_us_exp'*X_us_exp) / norm(F_us*x_est(:,k) - X_us,2)  + gamma * sign(x_est(:,k));

    % Calculate new x
    x_est(:,k+1) = x_est(:,k) - step_size*nabula;
    
    % Calculate error
    error = [error; norm(F_us*x_est(:,k) - X_us, 2)];
    
    if error(end) <= best_error
        best_x_est = x_est(:,k);
        best_error = error(end);
    end

    k = k + 1;
end
tEnd = cputime - tStart;
disp("CPU time since start of loop")
disp(tEnd);

disp("Stopping criterion:")
disp("norm(F_us*x_est(:,k) - X_us, 2) + gamma*norm(x_est,1) < 0.01")
disp(norm(F_us*x_est(:,k) - X_us, 2) + gamma*norm(x_est,1))
disp("Steps to get to stopping criterion:")
disp(k)

disp("Final error:") 
disp(norm(F_us*best_x_est - X_us, 2))

disp("Error with true vector")
disp(norm(best_x_est-x,2))

figure;
plot(error)
yscale("log")
title("Error")

figure;
plot(real(best_x_est))
title("Estimated x using gradient descent for constrained problem")
