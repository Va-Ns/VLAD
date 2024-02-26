function Codes = ProductQuantizationNV(VLAD_Representation,Options)

    arguments

        VLAD_Representation (:,:)      {mustBeNonempty}

        Options.numSubvectors          {mustBeInteger,mustBePositive,...
                                        mustBeNonzero} = 8

        Options.numCentroids           {mustBeInteger,mustBePositive,...
                                        mustBeNonzero} = 256
                
    end

    for numImages = 1 : size(VLAD_Representation,1)

        fprintf("Now in image: %d of %d\n",numImages,size(VLAD_Representation,1))

        Current_VLADImgRep = VLAD_Representation(numImages,:);
        

        %% Splitting Phase 

        SplittedVectors = SplittingPhase(Current_VLADImgRep,Options.numSubvectors);



    end





end