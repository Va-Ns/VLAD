%% Clean-up
clear;clc;close('all');
%% Initialize the parallel pool and set a steady random seed generator
delete(gcp('nocreate'))
maxWorkers = maxNumCompThreads;
disp("Maximum number of workers: " + maxWorkers);
pool=parpool(maxWorkers/2);
s = rng("default");
%% Get images directory and form the imageDatastore
fileLocation = uigetdir();
datastore = imageDatastore(fileLocation,"IncludeSubfolders",true, ...
    "LabelSource","foldernames");

%% Counting the number of labels 
initialLabels = countEachLabel(datastore);

splitDatastore = splitEachLabel(datastore,1/4);
newlabels = countEachLabel(splitDatastore);

[Trainds,Testds,training_labels,testing_labels] = ...
                   splitTheDatastore(splitDatastore,newlabels,"flag",true);

%% Generate SIFT descriptors using Dense SIFT.
train_features = denseSIFTVasilakis(Trainds);
test_features = denseSIFTVasilakis(Testds);

%% Formating the Dictionary and extracting the SIFT matrices for the sets
for k = 1: length(train_features)

        reset(train_features{k})

end

Dictionary = DictionaryFormationVasilakis(train_features);

%% Implementation of the VLAD 

U = VLADNV(Dictionary,train_features);

%% Product quantization

Codes = ProductQuantizationNV(U);