function varargout = splitTheDatastore(datastore,newlabels, ...
                                       optional)

%% Description
%  ------------------------------------------------------------------------
%  A function that splits the datastore into training and testing 
%  datastores based on user's input that decides whether the data should be 
%  processed to have equal number of labels or not. The optional variable 
%  opt gives the user the choise to either return the training and testing 
%  label tables with the count of the former or not.


% Inputs:
% -------------------------------------------------------------------------
% => datastore:     The imageDatastore that contains the raw data.
%
% => newlabels:     A variable that contains the labels that are extracted 
%                   from the initial datastore.
% => optional:      Char or string variable that takes two discrete values: 
%                   "Equal" or "Proceed". Based on one of the two, the 
%                   appropriate measures are taken to treat the splitting 
%                   procedure.

% Outputs: 
%
% => varargout:     As the name depicts, a variable argument output, that
%                   based on the option variable returns the training and 
%                   testing variables.If the optional variable's optional 
%                   flag is true, then the function returns the table of 
%                   training and testing datastores that contain both the 
%                   labels and the count of them for the user to process.
arguments (Input)

    datastore          {mustBeUnderlyingType(datastore, ...
                                             ['matlab.io.datastore.' ...
                                             'ImageDatastore'])} 

    newlabels          {mustBeNonempty} 

    optional.flag      {mustBeNumericOrLogical} = false
    
    

end


if numel(nargin) < 4


    [Trainds,Testds] = splitEachLabel(datastore,0.7,'randomized');
    trainlabelcount=countEachLabel(Trainds);
    testlabelcount=countEachLabel(Testds);


    if optional.flag
        varargout = {Trainds,Testds,trainlabelcount,testlabelcount};
    else
        varargout = {Trainds,Testds};
    end

else

    error("Too many function inputs")

end


end