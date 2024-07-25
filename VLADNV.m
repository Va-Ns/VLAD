function U_ij = VLADNV(Dictionary,features)

    % Reset the features ArrayDatastore
    for k = 1 : length(features)

        reset(features{k})
        
    end

    [k,d] = size(Dictionary);
    U_ij = zeros(size(features,1),k*d);

    for numImages = 1 : size(features,1)

        fprintf("Now in image: %d of %d\n",numImages,size(features,1))

        Current_ImgFeatData = read(features{numImages});
        Current_ImgFeatData = gpuArray(Current_ImgFeatData.data);
        
        % Find the nearest neighbors of the image in question
        [~,index] = pdist2(Dictionary,Current_ImgFeatData, ...
                                                'euclidean',"Smallest",1);
        % Compute the residual
        imageResiduals = Current_ImgFeatData - Dictionary(index,:);
        
        % Find the unique indices of the clusters that are present for the
        % current image. That means that when the local descriptors of the
        % image are taken separately, there's a chance some clusters to not
        % have any local descriptors
        uniqueElements = unique(index);
        
        % In order to counter the fact of not taking into account those clusters that don't have any
        % local descriptors, we create an array of zeros, that have the dimensionality of a k-by-d 
        % matrix and place the residuals to the corresponding indices of the matrix, while also 
        % leaving the clusters of zero contribution present. In this way, we can create a stable 
        % 1-by-k*d representation.
        ImageRepresentation = zeros(k, d);

        for cluster = uniqueElements

            uniqueElementscol = index == cluster;
            Residuals = sum(imageResiduals(uniqueElementscol,:),1);
            Residuals = fillmissing(Residuals/norm(Residuals),'constant',0);
            ImageRepresentation(cluster,:) = Residuals;

        end
        
        % We reshape the ImageRepresentation matrix into a vector and
        % transpose it to fit the line of U_ij.
        U_ij(numImages,:) = ImageRepresentation(:)';
    end
end