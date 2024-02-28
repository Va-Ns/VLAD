function Codes = ProductQuantizerNV(SplittedVectors,Options)

arguments

    SplittedVectors (:,:) {mustBeNonempty}
    
    Options.Clusters      {mustBeNonNan,mustBePositive,mustBePowerOfTwo} = 256

end

    Index = zeros(size(SplittedVectors,2),size(SplittedVectors,1));
    Centers = zeros(Options.Clusters,size(SplittedVectors,1));
    

    % Here every loop is a discrete subquantizer
    for subVector = 1 : size(SplittedVectors,1)
        
        [Index(:,subVector),Centers(:,subVector)] = ...
        kmeans(gpuArray(SplittedVectors(subVector,:)'),Options.Clusters);

    end
    
    Codes.Index = Index;
    Codes.Centers= Centers;
end