function SplittedVectors = SplittingPhase(Vector_Representation,numSubvectors)
    
    % Get the dimensions of the VLAD Representation 
    [~,VLAD_RepDim] = size(Vector_Representation);
    

    SubVectorDim = VLAD_RepDim / numSubvectors;

    fprintf("Creating %d subvectors of dimension: %d", numSubvectors, ...
                                                             SubVectorDim);

    SplittedVectors = reshape(Vector_Representation, SubVectorDim, []).';

end