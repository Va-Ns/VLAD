function Database = ProductQuantizationNV(VLAD_Representation,Options)

    arguments

        VLAD_Representation (:,:)      {mustBeNonempty}

        Options.numSubvectors          {mustBeInteger,mustBePositive,...
                                        mustBeNonzero} = 8

        Options.numCentroids           {mustBeInteger,mustBePositive,...
                                        mustBeNonzero,mustBePowerOfTwo} = 256
                
    end
    
    SubVectorDim = size(VLAD_Representation,2) / Options.numSubvectors;
    fprintf("Creating %d subvectors of dimension: %d\n", Options.numSubvectors, ...
                                                             SubVectorDim);

    for numImages = 1 : size(VLAD_Representation,1)

        fprintf("Now in image: %d of %d\n",numImages,size(VLAD_Representation,1))

        Current_VLADImgRep = VLAD_Representation(numImages,:);
        

        %% Splitting Phase 
    
        SplittedVectors = SplittingPhase(Current_VLADImgRep, ...
                                                    Options.numSubvectors);
        
        %% Product Quantization 

        Codes = ProductQuantizerNV(SplittedVectors);
    	Database(numImages).Index = Codes.Index;
        Database(numImages).Centers = Codes.Centers;
        

    end


end