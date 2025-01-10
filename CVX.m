clear all
close all
clc
load("cs.mat")

x_true = x;

figure
plot(x);
title("true data vector")

gamma = 0.1;

% Show the sparse data and data mask
figure;
plot(sampling_mask);
title("Sampling mask");

figure
plot(x);
title("true data vector")

figure
plot(real(F_us*x))
title("Sampling_mask*x")

figure;
plot(real(X_us));
title("X_us");
cvx_begin
    cvx_precision best
    cvx_solver_settings('dumpfile', 'Test')
    variable x_est(n)
    minimize( norm(F_us*x_est-X_us, 2) + gamma*norm(x_est,1) )
    subject to
        x_est(n) >= 0
cvx_end

load("Test.mat")

figure;
plot(x_est)
title("x estimated using CVX")

disp("Final error:") 
disp(norm(F_us*x_est - X_us, 2))

disp("Error with true vector")
disp(norm(x_est-x_true,2))
