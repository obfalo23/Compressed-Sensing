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
alpha = 0.01;
beta = 0.5;
gamma = 0.1;

disp("Initial error:")
disp(norm(F_us*x_est(:,1) - X_us, 2))
best_error = 1000;
k = 1;
max_steps = 5000;
tStart = cputime;
diff_x = 0;
while norm(F_us*x_est(:,k) - X_us, 2) > epsilon && k < max_steps
    
    % Calculate using backtracking line search
    t = 1;
    while (norm(F_us*(xest(:,k) + t*diff_x)-X_us, 2) + gamma * sign((xest(:,k) + t*diff_x))) > (norm(F_us*xest(:,k)-X_us, 2) + gamma * sign(xest(:,k) + t*diff_x)) + alpha*t*nabula*diff_x) 
        t = beta*t;
    end
    
    % Calculate first derivatives (direction)
    nabula = 2*F_us'*F_us*x_est(:,k) - 2*F_us'*X_us + gamma * sign(x_est(:,k));
    Hessian = 2*F_us'*F_us + gamma;
    
    % Update estimate of x
    x_est(:,k+1) = x_est(:,k) - (1/Hessian)*nabula;
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
title("Estimated x")
