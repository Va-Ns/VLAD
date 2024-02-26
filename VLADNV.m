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

        [~,index] = pdist2(Dictionary,Current_ImgFeatData, ...
                'euclidean',"Smallest",1);

        imageResiduals = Current_ImgFeatData - Dictionary(index,:);

        uniqueElements = unique(index);

        ImageRepresentation = zeros(k, d);

        for cluster = uniqueElements
            uniqueElementscol = index == cluster;
            Residuals = sum(imageResiduals(uniqueElementscol,:),1);
            Residuals = Residuals/norm(Residuals);
            ImageRepresentation(cluster,:) = Residuals;
        end

        U_ij(numImages,:) = ImageRepresentation(:)';
    end
end