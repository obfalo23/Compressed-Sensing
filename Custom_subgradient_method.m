clear all
close all
clc
load("cs.mat")

% Show the sparse data and data mask
figure
plot(x);
title("true data vector")

figure;
plot(real(X_us));
title("X us / F_us*x");

% Create the initial x_est
x_est = zeros(128,1000);

error = double.empty;

step_size = 1;
epsilon = 0.01; % Stop criterion

disp("Initial error:")
disp(norm(F_us*x_est(:,1) - X_us, 2))
best_error = 1000;
k = 1;
max_steps = 5000;
tStart = cputime;
while norm(F_us*x_est(:,k) - X_us, 2) > epsilon && k < max_steps
    % Calculate step size
    step_size = 10/k;

    % Check if the solution is holding to the constraint
    disp(norm(x_est(:,k), 1))
    if k <= 1000
        feas_thres = 1 + 2*k/max_steps;
    else
        feas_thres = 3;
    end

    if norm(x_est(:,k), 1) > feas_thres
        nabula = sign(x_est(:,k));
    else
        % Calculate first derivatives (direction)
        nabula = real(2*F_us'*F_us*x_est(:,k) - 2*F_us'*X_us);
        Hessian = real(2*F_us*F_us);
    end

    x_est(:,k+1) = abs(x_est(:,k) - step_size*nabula);
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
title("Estimated x using Custom gradient descent")
