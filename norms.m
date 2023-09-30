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
data = input('Enter the path to the file containing core data: ', 's');

% Load the precomputed decomposition factors and core tensor
load(data)

N = ndims(sparse_tensor);
% U = cp_data.U;
U = c_core.U;
maxiters = 100;
dimorder = [3, 2, 1];
printitn = 1;
fitchangetol = 1e-4;

% Initialize an array to store the fits of each slice
slice_fits = zeros(tensor_size(1), 1);

% Loop over each slice and calculate the fit
for k = 1:tensor_size(1)
    % Extract the k-th slice from the tensor
    slice_tensor = sparse_tensor(:, :, k);
    
    % Reconstruct the k-th slice using the factor matrices A, B, and C
    reconstructed_slice = U{1} * diag(c_core.lambda) * diag(U{3}(k, :)) * U{2}';
    
    % Calculate the fit of the k-th slice (e.g., Frobenius norm)
    fit = norm(double(slice_tensor) - reconstructed_slice, 'fro') / norm(slice_tensor, 'fro');

    disp("Fit for slice %d: %d", k, fit)
    
    % Store the fit of the k-th slice
    slice_fits(k) = fit;
end