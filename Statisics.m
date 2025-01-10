clear all
close all
clc
load("cs.mat")
x_true = x;

% Start the statistics routine here
N = n; % Data size
epsilon = 1e-15; % Stop criterion
K = 5000;
gamma = 0.1;

% Statistics parameters and stucts
Methods = 5;
P = 50;
error_struct = zeros(Methods,P,K);
iteration_struct = zeros(Methods, P);
cpuTime_struct = zeros(Methods, P);
best_error_struct = zeros(Methods, P);
true_data_error_struct = zeros(Methods, P);

for p = 1:P
    %% CVX
    cvx_begin
        cvx_precision best
        cvx_solver_settings('dumpfile', 'Test')
        variable x_est(n)
        minimize( norm(F_us*x_est-X_us, 2) + gamma*norm(x_est,1) )
        subject to
            x_est(n) >= 0
    cvx_end
    load("Test.mat")

    % Save statistics
    error_struct(1,p,1) = norm(F_us*x_est - X_us, 2);
    best_error_struct(1,p) = error_struct(1,p,1);
    iteration_struct(1,p) = cvx_slvitr;
    cpuTime_struct(1,p) = cvx_cputime;
    true_data_error_struct(1, p) = norm(x_est-x_true,2);

    %% Subgradient_descent_max_x_l1 and linear Step_size
    
    % Init
    x_est = zeros(128,K);
    step_size = 1;
    error = zeros(K,1);
    error(1) = norm(F_us*x_est(:,1) - X_us, 2);
    best_error = error(1); % tracking the best error
    
    k = 1;
    tStart = cputime;
    while norm(F_us*x_est(:,k) - X_us, 2) > epsilon && k < K
        % Calculate step size
        step_size = 10/k;
    
        % Check if the solution is holding to the constraint
        if k <= 1000
            feas_thres = 1 + 2*k/K;
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
        
        % Calculate error and decide on best error
        error(k) = norm(F_us*x_est(:,k+1) - X_us, 2);
        if(norm(F_us*x_est(:,k) - X_us, 2) < epsilon) && iteration_struct(2,p) <= 1
            if error(end) <= best_error
                best_x_est = x_est(:,k);
                best_error = error(k);
            end
            tEnd = cputime - tStart;
            cpuTime_struct(2,p) = tEnd;
            iteration_struct(2,p) = k;
        end
    
        k = k + 1;
    end
    tEnd = cputime - tStart;
    best_x_est = x_est(:,k-1);

    % Save statistics
    error_struct(2,p,:) = error(:);
    best_error_struct(2,p) = best_error;
    true_data_error_struct(2, p) = norm(best_x_est-x_true,2);
    
    if iteration_struct(2,p) == 0
        iteration_struct(2,p) = k;
        cpuTime_struct(2,p) = tEnd;
    end

    %% Sub gradient linear step size absolute x_est (basis persuit denoising)

    % Use a twice as tall expanded matrix to be able to differentiate an
    % otherwise complex matrix
    F_us_exp = [real(F_us);imag(F_us)];
    X_us_exp = [real(X_us);imag(X_us)];
    
    % Init
    x_est = zeros(128,K);
    step_size = 1;
    error = zeros(K,1);
    error(1) = norm(F_us*x_est(:,1) - X_us, 2);
    best_error = error(1); % tracking the best error
    
    % Descent till stop criterion is met on l2 error
    k = 1;
    tStart = cputime;
    while k < K
        % Calculate first derivatives (direction)
        nabula = (2*(F_us_exp'*F_us_exp)*x_est(:,k) - 2*F_us_exp'*X_us_exp) / norm(F_us*x_est(:,k) - X_us,2)  + gamma * sign(x_est(:,k));
        
        % Calculate step size
        step_size = 0.4/(k+1);
    
        % Calculate new x from absolute of next step to remove negative part
        x_est(:,k+1) = abs(x_est(:,k) - step_size*nabula);
        
        % Calculate error and decide on best error
        error(k) = norm(F_us*x_est(:,k+1) - X_us, 2);
        if(norm(F_us*x_est(:,k) - X_us, 2) < epsilon) && iteration_struct(3,p) <= 1
            if error(end) <= best_error
                best_x_est = x_est(:,k);
                best_error = error(k);
            end
            tEnd = cputime - tStart;
            cpuTime_struct(3,p) = tEnd;
            iteration_struct(3,p) = k;
        end

        k = k + 1;
    end
    tEnd = cputime - tStart;
    best_x_est = x_est(:,k-1);

    % Save statistics
    error_struct(3,p,:) = error(:);
    best_error_struct(3,p) = best_error;
    true_data_error_struct(3, p) = norm(best_x_est-x_true,2);

    if iteration_struct(3,p) == 0
        iteration_struct(3,p) = k;
        cpuTime_struct(3,p) = tEnd;
    end
    
    %% Projected subgradient method
    
    % Value to limit the step size
    step_size_exp_parameter = 70;

    % Use a twice as tall expanded matrix to be able to differentiate an
    % otherwise complex matrix
    F_us_exp = [real(F_us);imag(F_us)];
    X_us_exp = [real(X_us);imag(X_us)];
    
    % Init
    x_est = zeros(128,K);
    step_size = 1;
    error = zeros(K,1);
    error(1) = norm(F_us*x_est(:,1) - X_us, 2);
    best_error = error(1); % tracking the best error

    % Descent till stop criterion is met on l2 error
    k = 1;
    tStart = cputime;
    while k < K
        % Calculate first derivatives (direction)
        nabula = (2*(F_us_exp'*F_us_exp)*x_est(:,k) - 2*F_us_exp'*X_us_exp) / norm(F_us*x_est(:,k) - X_us,2)  + gamma * sign(x_est(:,k));
        
        % Calculate step size with decreasing exponential
        step_size = exp(-(k)/step_size_exp_parameter)/(k+1);
    
        % Subgradient descent
        x_est(:,k+1) = x_est(:,k) - step_size*nabula(:);
        
        % Enforcing non-negativity constrained by projection
        x_est(:,k+1) = max(0, x_est(:,k+1));
    
        % Calculate error and decide on best error
        error(k) = norm(F_us*x_est(:,k+1) - X_us, 2);
        if(norm(F_us*x_est(:,k) - X_us, 2) < epsilon) && iteration_struct(4,p) <= 1
            if error(end) <= best_error
                best_x_est = x_est(:,k);
                best_error = error(k);
            end
            tEnd = cputime - tStart;
            cpuTime_struct(4,p) = tEnd;
            iteration_struct(4,p) = k;
        end
    
        k = k + 1;
    end
    tEnd = cputime - tStart;
    
    % Save statistics
    error_struct(4,p,:) = error(:);
    best_error_struct(4,p) = best_error;
    true_data_error_struct(4, p) = norm(best_x_est-x_true,2);
    
    if iteration_struct(4,p) == 0
        iteration_struct(4,p) = k;
        cpuTime_struct(4,p) = tEnd;
    end

    %% Custom method
    
    % Value to limit the step size
    step_size_exp_parameter = 50;

    % Use a twice as tall expanded matrix to be able to differentiate an
    % otherwise complex matrix
    F_us_exp = [real(F_us);imag(F_us)];
    X_us_exp = [real(X_us);imag(X_us)];
    
    % Init
    x_est = zeros(128,K);
    step_size = 1;
    error = zeros(K,1);
    error(1) = norm(F_us*x_est(:,1) - X_us, 2);
    best_error = error(1); % tracking the best error

    % Descent till stop criterion is met on l2 error
    k = 1;
    tStart = cputime;
    while k < K
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
        if(norm(F_us*x_est(:,k) - X_us, 2) < epsilon) && iteration_struct(5,p) <= 1
            if error(end) <= best_error
                best_x_est = x_est(:,k);
                best_error = error(k);
            end
            tEnd = cputime - tStart;
            cpuTime_struct(5,p) = tEnd;
            iteration_struct(5,p) = k;
        end
    
        k = k + 1;
    end
    tEnd = cputime - tStart;
    
    % Save statistics
    error_struct(5,p,:) = error(:);
    best_error_struct(5,p) = best_error;
    true_data_error_struct(5, p) = norm(best_x_est-x_true,2);

    if iteration_struct(5,p) == 0
        iteration_struct(5,p) = k;
        cpuTime_struct(5,p) = tEnd;
    end

    %% loop debug
    p % print progress
end

%% CPU time
disp("Mean CPUtime CVX")
disp(mean(cpuTime_struct(1,:)))
disp("Mean CPUtime max_l1")
disp(mean(cpuTime_struct(2,:)))
disp("Mean CPUtime basis")
disp(mean(cpuTime_struct(3,:)))
disp("Mean CPUtime proj_sub")
disp(mean(cpuTime_struct(4,:)))
disp("Mean CPUtime Custom")
disp(mean(cpuTime_struct(5,:)))

% And Standard deviation
disp("std dev CPUtime CVX")
disp(std(cpuTime_struct(1,:)))
disp("std dev CPUtime max_l1")
disp(std(cpuTime_struct(2,:)))
disp("std dev CPUtime basis")
disp(std(cpuTime_struct(3,:)))
disp("std dev CPUtime proj_sub")
disp(std(cpuTime_struct(4,:)))
disp("std dev CPUtime Custom")
disp(std(cpuTime_struct(5,:)))

%% Iteration
disp("Mean iterations CVX")
disp(mean(iteration_struct(1,:)))
disp("Mean iterations max_l1")
disp(mean(iteration_struct(2,:)))
disp("Mean iterations basis")
disp(mean(iteration_struct(3,:)))
disp("Mean iterations proj_sub")
disp(mean(iteration_struct(4,:)))
disp("Mean iterations Custom")
disp(mean(iteration_struct(5,:)))

%% True data error
disp("Mean true data error CVX")
disp(mean(true_data_error_struct(1,:)))
disp("Mean true data error max_l1")
disp(mean(true_data_error_struct(2,:)))
disp("Mean true data error basis")
disp(mean(true_data_error_struct(3,:)))
disp("Mean true data error proj_sub")
disp(mean(true_data_error_struct(4,:)))
disp("Mean true data error Custom")
disp(mean(true_data_error_struct(5,:)))

% And Standard deviation
disp("Std dev true data error CVX")
disp(std(true_data_error_struct(1,:)))
disp("Std dev true data error max_l1")
disp(std(true_data_error_struct(2,:)))
disp("Std dev true data error basis")
disp(std(true_data_error_struct(3,:)))
disp("Std dev true data error proj_sub")
disp(std(true_data_error_struct(4,:)))
disp("Std dev true data error Custom")
disp(std(true_data_error_struct(5,:)))

%% Plotting
plot_CVX = reshape(mean(error_struct(1,:,:),2),K,1);
plot_max_l1 = reshape(mean(error_struct(2,:,:),2),K,1);
plot_basis = reshape(mean(error_struct(3,:,:),2),K,1);
plot_proj_sub = reshape(mean(error_struct(4,:,:),2),K,1);
plot_Custom = reshape(mean(error_struct(5,:,:),2),K,1);

iterations = linspace(1,K,K);
figure

plot(iterations,plot_max_l1, 'DisplayName','max l1');
hold on;
plot(iterations,plot_basis, 'DisplayName','basis');
hold on;
plot(iterations,plot_proj_sub, 'DisplayName','proj sub');
hold on;
plot(iterations,plot_Custom, 'DisplayName','Custom');
hold on;

yscale("log")
grid("on")
ylabel("Error")
xlabel("Time [s]")
title("Mean convergence over 50 runs")
legend("AutoUpdate","on")
