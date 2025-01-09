clear all
close all
load("cs.mat")

gamma = 0.1
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
    variable x_est(n)
    minimize( norm(F_us*x_est-X_us, 2) + gamma*norm(x_est,1) )
cvx_end

figure;
plot(x_est)
title("x estimated using CVX")

disp("Final error:") 
disp(norm(F_us*x_est - X_us, 2))

disp("Error with true vector")
disp(norm(x_est-x,2))

% Determine sparsity
% non_zero_array = find([X_us(1:end-1)]);
% K = size(non_zero_array)
% 
% Z = [diff([1,non_zero_array(:).'+1])-1;X_us(non_zero_array(:).',1).'].'
