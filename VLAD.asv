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

%% Formation of the Dictionary and extracting the SIFT matrices for the sets
for k = 1: length(train_features)

        reset(train_features{k})

end

Dictionary = DictionaryFormationVasilakis(train_features);

%% Implementation of the VLAD 

U_Training = VLADNV(Dictionary,train_features);
U_Testing = VLADNV(Dictionary,test_features);

%% Product quantization

% tic
% Database = ProductQuantizationNV(U_Training);
% Product_quantization_time = toc


%%

t = templateSVM('SaveSupportVectors',true,'Type','classification');
[Model1,HyperparameterOptimizationResults] = fitcecoc(gpuArray(U_Training), ...
    Trainds.Labels,"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions',struct('Holdout',0.1, ...
    'MaxObjectiveEvaluations',100));

%%

[predictedLabels, scores]= predict(Model1,U_Testing);

confusionMatrix = confusionmat(Testds.Labels,predictedLabels);

% Υπολογισμός ακρίβειας
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

