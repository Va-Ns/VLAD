function [bestCentroids, bestCost, timeElapsed] = miniBatchKMeansNV(data,Options)
    

arguments 
    
    data                       {mustBeNonempty}

    Options.numClusters        {mustBePositive,mustBeInteger} = 200

    Options.maxIter            {mustBeInteger,...
                                mustBeInRange(Options.maxIter,1,1e9)} = 50

    Options.replicates         {mustBeInteger,...
                                mustBeInRange(Options.replicates,1,1e9)} = 10

    Options.batchSize          {mustBeInteger,mustBePositive} = 1000

end

    batchSize = Options.batchSize;
    maxIter = Options.maxIter;
    replicates = Options.replicates;
    numClusters = Options.numClusters;

    fprintf("Using: \n\n Batch Size = %d \n\n maxIter =%d \n\n " + ...
            "Replicates =%d \n\n numClusters = %d\n\n", ...
            batchSize,maxIter,replicates,numClusters);

    % Ensure data is in single precision (float32)
    dataGPU = gpuArray(single(data));
 
    % Initialize shared variables (persistent variables in the workers' workspace)
    bestCost = Inf;
    bestCentroids = zeros(numClusters, size(dataGPU, 2), 'single');

    % Randomly shuffle the data
    shuffledData = dataGPU(randperm(size(data, 1)), :);

    % Initialize centroids using kmeans++
    currentCentroids = datasample(shuffledData, numClusters, 'Replace', ...
                                                                    false);

    % Perform mini-batch k-means
    for replicate = 1:replicates

        for iter = 1:maxIter
            
            % Select a mini-batch
            startIdx = (iter - 1) * batchSize + 1;
            endIdx = min(iter * batchSize, size(shuffledData, 1));
            miniBatch = shuffledData(startIdx:endIdx, :);

            % Update centroids using the mini-batch
            [~, currentCentroids,sumd] = kmeans(miniBatch, numClusters, 'Start', ...
                                           currentCentroids);

            fprintf("---------------------------------------------------\n" + ...
                    "Now in Replicate: %d | Iteration: %d\n" + ...
                    "---------------------------------------------------\n", ...
                replicate,iter);
        end
        
        totalCost = sum(sumd);

        % Update the shared variables
        if totalCost < bestCost

            bestCentroids = currentCentroids;
            bestCost = totalCost;
            
        end
    end

    timeElapsed = toc;
        
end
