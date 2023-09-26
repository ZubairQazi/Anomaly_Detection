addpath(fullfile('~', 'tensor_toolbox'));

data_path = input('Enter the path to the file containing tensor data: ', 's');

% Load data from the .mat file
data = load(data_path);

% Extract the data and display tensor size
indices = data.indices;
values = double(data.values(:)); % Column vector
tensor_size = data.size;

disp('Tensor size:');
disp(tensor_size);

% Convert indices to 1-based indexing
indices = indices + 1;

% Create the sparse tensor using sptensor
sparse_tensor = sptensor(indices, values, tensor_size);

% Ask the user for the location of the decomposition files
factors = input('Enter the path to the file containing factor data: ', 's');
core = input('Enter the path to the file containing core data: ', 's');

% Load the precomputed decomposition factors and core tensor
load(factors); % Load decomposition factors
load(core)

R = size(c_factors{2}, 2);
N = ndims(sparse_tensor);
U = c_factors;
maxiters = 100;
dimorder = [1, 2, 3];
printitn = 1;
fitchangetol = 1e-4;
fit = 0;
normX = norm(sparse_tensor);

%% Main Loop: Iterate until convergence
UtU = zeros(R,R,N);
for n = 1:N
    if ~isempty(U{n})
        UtU(:,:,n) = U{n}'*U{n};
    end
end
    
for iter = 1:maxiters
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    for n = dimorder(1:end)
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        Unew = mttkrp(sparse_tensor,U,n);
        
        % Compute the matrix of coefficients for linear system
        % disp(size(Y)); disp(size(Unew)); disp(size(UtU))
        Y = prod(UtU(:,:,[1:n-1 n+1:N]),3);
        Unew = Unew / Y;
        if issparse(Unew)
            Unew = full(Unew);   % for the case R=1
        end
        
        % Normalize each vector to prevent singularities in coefmatrix
        if iter == 1
            lambda = sqrt(sum(Unew.^2,1))'; %2-norm
        else
            lambda = max( max(abs(Unew),[],1), 1 )'; %max-norm
        end
        
        Unew = bsxfun(@rdivide, Unew, lambda');
        U{n} = Unew;
        UtU(:,:,n) = U{n}'*U{n};
    end
    
    P = ktensor(lambda,U);

    if normX == 0
        fit = norm(P)^2 - 2 * innerprod(sparse_tensor,P);
    else
        normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(sparse_tensor,P) );
        fit = 1 - (normresidual / normX); %fraction explained by model
    end

    fitchange = abs(fitold - fit);
    
    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        flag = 0;
    else
        flag = 1;
    end
    
    if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
    end
    
    % Check for convergence
    if (flag == 0)
        break;
    end

end 

disp(fit)