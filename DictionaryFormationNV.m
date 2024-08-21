function [Dictionary] = DictionaryFormationNV(train_features,Options)

    arguments (Input)

    train_features          {mustBeA(train_features,"cell")}

    

    
    Options.Centers         {mustBeInteger,mustBePositive} = 200
    

    end
    
    % Preallocation of the size of the Centers variable as it will be of
    % use in the use of k-means 
    Dictionary = zeros(Options.Centers,128);

    % Due to the variable size of the images, the SIFT descriptors matrix
    % is not trivial to be computed in order to preallocate it. 
    training_SIFT_matrix= [];

    tic

    % Extract the SIFT descriptor data as they are stored as ArrayDatastore
    % variables and create a collective matrix that contains all the
    % descriptors of all the training images
    
    for i = 1:length(train_features)
    
        feature_data = read(train_features{i});

        training_SIFT_matrix = [training_SIFT_matrix;feature_data.data];
        
    end
    
    
    % Get the size of the SIFT matrix
    [numKeypoints DescDim] = size(training_SIFT_matrix);

    % The number of Centers provided by the user must not exceed the total
    % number of keypoints as they are formed by the collective matrix of
    % SIFT descriptors, e.g. here SIFT_matrix's first dimension.
    
    validateattributes(Options.Centers,'numeric',{'<',numKeypoints})
    
    [bestCentroids, bestCost, timeElapsed] = miniBatchKMeansNV(training_SIFT_matrix);

    % Display results
    disp('Best cost:')
    disp(bestCost);
    disp(['Elapsed Time: ' num2str(timeElapsed) ' seconds']);
    
    Dictionary = bestCentroids;

end