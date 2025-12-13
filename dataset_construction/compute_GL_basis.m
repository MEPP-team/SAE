% Compute Graph Laplacian basis for smpl_faces.txt triangulation
% This script reads the smpl_faces.txt file and computes the Graph Laplacian eigenbasis

clc; clear all; close all;

% Parameters
triv_file = 'smpl_faces.txt';
num_eigenvectors = 6890;  % Number of eigenvectors to compute

fprintf('Computing Graph Laplacian Basis\n');

% Load triangulation
fprintf('Loading triangulation from %s...\n', triv_file);
TRIV = load(triv_file);

% TRIV is 0-indexed (from Python), convert to 1-indexed (MATLAB)
TRIV = TRIV + 1;

num_faces = size(TRIV, 1);
num_vertices = max(TRIV(:));

fprintf('  Number of faces: %d\n', num_faces);
fprintf('  Number of vertices: %d\n\n', num_vertices);

% Build adjacency matrix
fprintf('Building adjacency matrix...\n');
A = adjacency_matrix(TRIV, num_vertices);
fprintf('  Adjacency matrix size: %dx%d\n', size(A, 1), size(A, 2));
fprintf('  Number of edges: %d\n\n', nnz(A) / 2);

% Compute Graph Laplacian
fprintf('Computing Graph Laplacian matrix...\n');
% L = diag(sum(A)) - A  (combinatorial Laplacian)
degree = full(sum(A, 2));  % Row sums (degree of each vertex)
L = diag(degree) - A;

% Convert to sparse format
L = sparse(L);
fprintf('  Laplacian matrix size: %dx%d\n', size(L, 1), size(L, 2));
fprintf('  Laplacian is sparse: %d non-zeros\n\n', nnz(L));

% Compute eigenvectors
num_eigs = min(num_eigenvectors, num_vertices);
fprintf('Computing %d smallest eigenpairs (using shift-invert mode)...\n', num_eigs);

tic;
try
    [evecs, evals] = eigs(L, num_eigs, -1e-6, 'sm');
catch ME
    % Fallback: add small perturbation
    L_pert = L + 1e-8 * speye(num_vertices);
    [evecs, evals] = eigs(L_pert, num_eigs, 'sm');
end
elapsed_time = toc;

% Extract eigenvalues from diagonal matrix
evals = diag(evals);

% Sort eigenvalues and eigenvectors in ascending order
[evals, order] = sort(evals, 'ascend');
evecs = evecs(:, order);

% Set first eigenvalue to exactly 0 if it's very small (numerical error)
if evals(1) < 0 && abs(evals(1)) < 1e-7
    fprintf('Setting first eigenvalue to 0 (was %.2e)...\n', evals(1));
    evals(1) = 0;
end

% Normalize eigenvalues to [0, 1]
if evals(end) ~= 0
    evals = evals / evals(end);
end

% Normalize evecs
evecs = normc(evecs);

fprintf('  Computed %d eigenvectors in %.2f seconds\n', size(evecs, 2), elapsed_time);
fprintf('  Eigenvalue range: [%.6f, %.6f]\n', evals(1), evals(end));
fprintf('  First 10 eigenvalues: ');
fprintf('%.6f ', evals(1:min(10, length(evals))));
fprintf('\n\n');

% Save results
fprintf('Saving results...\n');

% Save in ASCII format for transfer to python
evecs_txt = sprintf('evecs_GL_%d.txt', num_eigs);
save(evecs_txt, 'evecs', '-ascii');
fprintf('    Eigenvectors saved to %s (ASCII format)\n', evecs_txt);

fprintf('Graph Laplacian basis computed successfully:\n');
fprintf('  Triangulation: %s (%d vertices, %d faces)\n', triv_file, num_vertices, num_faces);
fprintf('  Number of eigenvectors: %d\n', size(evecs, 2));
fprintf('  Eigenvalue range: [%.6f, %.6f]\n', evals(1), evals(end));
fprintf('  Computation time: %.2f seconds\n', elapsed_time);
fprintf('\nDone!\n');

%% Helper function: Build adjacency matrix from triangulation
function A = adjacency_matrix(TRIV, num_vertices)
    % Build adjacency matrix from triangulation=
    %
    % Args:
    %   TRIV: Triangle connectivity (num_faces x 3), 1-indexed
    %   num_vertices: Number of vertices in the mesh
    %
    % Returns:
    %   A: Sparse adjacency matrix (num_vertices x num_vertices)
    
    % Create edges from faces (each face has 3 edges)
    edges = [
        TRIV(:, 1), TRIV(:, 2);  % Edge 1-2
        TRIV(:, 2), TRIV(:, 3);  % Edge 2-3
        TRIV(:, 3), TRIV(:, 1)   % Edge 3-1
    ];
    
    % Create bidirectional edges (i->j and j->i)
    rows = [edges(:, 1); edges(:, 2)];
    cols = [edges(:, 2); edges(:, 1)];
    data = ones(length(rows), 1);
    
    % Build sparse adjacency matrix
    A = sparse(rows, cols, data, num_vertices, num_vertices);
    
    % Remove duplicates by converting to binary (adjacency is 0 or 1)
    A = (A > 0);
    
    % Convert back to double for numerical operations
    A = double(A);
end