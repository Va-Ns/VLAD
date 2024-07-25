clc;
for i = 1:length(train_features)
    reset(train_features{i});
end

descriptor_1 = read(train_features{1});
descriptor_2 = read(train_features{2});
descriptor_1 = double(descriptor_1.data);
descriptor_2 = double(descriptor_2.data);

% Combine descriptors from both images
descriptors = [descriptor_1; descriptor_2];

% Apply PCA to reduce dimensionality to 2
[coeff, score, ~] = pca(descriptors);
reduced_descriptors = score(:, 1:2);

% Cluster the reduced descriptors using k-means
numClusters = 200;
[idx, centroids] = kmeans(reduced_descriptors, numClusters);

% Calculate residuals (Euclidean distance between descriptor and its centroid)
residuals = zeros(size(reduced_descriptors, 1), 1);
for i = 1:size(reduced_descriptors, 1)
    residuals(i) = norm(reduced_descriptors(i, :) - centroids(idx(i), :));
end

% Plot residuals
figure;
scatter(reduced_descriptors(:, 1), reduced_descriptors(:, 2), 50, residuals, 'filled');
colorbar;
title('Residuals between Descriptors and Cluster Centroids');
xlabel('Descriptor Dim 1');
ylabel('Descriptor Dim 2');
grid on;

% Plot Voronoi diagram
figure;
voronoi(centroids(:, 1), centroids(:, 2));
hold on;
scatter(reduced_descriptors(:, 1), reduced_descriptors(:, 2), 50, idx, 'filled');
scatter(centroids(:, 1), centroids(:, 2), 100, 'kx');
title('Voronoi Diagram of Clusters');
xlabel('Descriptor Dim 1');
ylabel('Descriptor Dim 2');
grid on;
hold off;

% Plot residuals on Voronoi diagram
figure;
voronoi(centroids(:, 1), centroids(:, 2));
hold on;
scatter(reduced_descriptors(:, 1), reduced_descriptors(:, 2), 50, residuals, 'filled');
scatter(centroids(:, 1), centroids(:, 2), 100, 'kx');
colorbar;
title('Voronoi Diagram with Residuals');
xlabel('Descriptor Dim 1');
ylabel('Descriptor Dim 2');
grid on;
hold off;

% Select two Voronoi cells to focus on
selected_cells = [1, 2]; % Indices of the selected Voronoi cells

% Create a new figure for the selected Voronoi cells
figure;
voronoi(centroids(:, 1), centroids(:, 2));
hold on;

% Plot the selected Voronoi cells with corresponding descriptors and residuals
colors = lines(length(selected_cells));
for j = 1:length(selected_cells)
    cell_idx = selected_cells(j);
    descriptors_in_cell = find(idx == cell_idx);
    
    % Plot descriptors in the selected cell
    scatter(reduced_descriptors(descriptors_in_cell, 1), reduced_descriptors(descriptors_in_cell, 2), 50, colors(j, :), 'filled');
    
    % Plot the center of the selected cell
    scatter(centroids(cell_idx, 1), centroids(cell_idx, 2), 100, 'kx');
    
    % Plot residuals as dotted lines
    for k = 1:length(descriptors_in_cell)
        descriptor_idx = descriptors_in_cell(k);
        plot([centroids(cell_idx, 1), reduced_descriptors(descriptor_idx, 1)], ...
             [centroids(cell_idx, 2), reduced_descriptors(descriptor_idx, 2)], 'Color', colors(j, :), 'LineStyle', '--');
    end
end

title('Selected Voronoi Cells with Descriptors and Residuals');
xlabel('Descriptor Dim 1');
ylabel('Descriptor Dim 2');
grid on;
hold off;