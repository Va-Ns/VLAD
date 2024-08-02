%% Clean-up
clear; clc; close('all');

%% Initialize the parallel pool and set a steady random seed generator
% delete(gcp('nocreate'))
% maxWorkers = maxNumCompThreads;
% disp("Maximum number of workers: " + maxWorkers);
% pool = parpool(maxWorkers / 2);
% s = rng("default");

%% Get images directory and form the imageDatastore
fileLocation = uigetdir();
datastore = imageDatastore(fileLocation, "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

%% Counting the number of labels 
initialLabels = countEachLabel(datastore);

%% Run the experiment 10 times
numExperiments = 10;
accuracies = zeros(numExperiments, 4); % Store accuracies for each optimization

for expIdx = 1:numExperiments
    disp("Running experiment " + expIdx);

    splitDatastore = splitEachLabel(datastore, 1/4);
    newlabels = countEachLabel(splitDatastore);

    [Trainds, Testds, training_labels, testing_labels] = ...
                       splitTheDatastore(splitDatastore, newlabels, "flag", true);

    %% Generate SIFT descriptors using Dense SIFT.
    train_features = denseSIFTNV(Trainds);
    test_features = denseSIFTNV(Testds);

    %% Formation of the Dictionary and extracting the SIFT matrices for the sets
    for k = 1:length(train_features)
        reset(train_features{k});
    end

    Dictionary = DictionaryFormationNV(train_features);

    %% Implementation of the VLAD 
    U_Training = VLADNV(Dictionary, train_features);
    U_Testing = VLADNV(Dictionary, test_features);

    %% Train the SVM model with different hyperparameter optimizations
    t = templateSVM('SaveSupportVectors', true, 'Type', 'classification');

    % Optimize BoxConstraint
    Model1 = fitcecoc(gpuArray(U_Training), Trainds.Labels, "Learners", t, "Coding", "onevsall", ...
        'OptimizeHyperparameters', {'BoxConstraint'}, ...
        'HyperparameterOptimizationOptions', struct('Holdout', 0.1, 'MaxObjectiveEvaluations', 100, ...
        "ShowPlots",false));
    [predictedLabels, ~] = predict(Model1, U_Testing);
    confusionMatrix = confusionmat(Testds.Labels, predictedLabels);
    accuracies(expIdx, 1) = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

    % Optimize BoxConstraint and KernelScale
    Model2 = fitcecoc(gpuArray(U_Training), Trainds.Labels, "Learners", t, "Coding", "onevsall", ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct('Holdout', 0.1, 'MaxObjectiveEvaluations', 100, ...
        "ShowPlots",false));
    [predictedLabels, ~] = predict(Model2, U_Testing);
    confusionMatrix = confusionmat(Testds.Labels, predictedLabels);
    accuracies(expIdx, 2) = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

    % Optimize all parameters
    Model3 = fitcecoc(gpuArray(U_Training), Trainds.Labels, "Learners", t, "Coding", "onevsall", ...
        'OptimizeHyperparameters', 'all', ...
        'HyperparameterOptimizationOptions', struct('Holdout', 0.1, 'MaxObjectiveEvaluations', 100, ...
        "ShowPlots",false));
    [predictedLabels, ~] = predict(Model3, U_Testing);
    confusionMatrix = confusionmat(Testds.Labels, predictedLabels);
    accuracies(expIdx, 3) = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

    % No optimization (baseline)
    Model4 = fitcecoc(gpuArray(U_Training), Trainds.Labels, "Learners", t, "Coding", "onevsall");
    [predictedLabels, ~] = predict(Model4, U_Testing);
    confusionMatrix = confusionmat(Testds.Labels, predictedLabels);
    accuracies(expIdx, 4) = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
end

%% Calculate and display the mean accuracy
meanAccuracies = mean(accuracies);
bestAccuracies = max(accuracies, [], 2);
meanBestAccuracy = mean(bestAccuracies);

disp("Mean accuracies for each optimization:");
disp("BoxConstraint: " + meanAccuracies(1));
disp("BoxConstraint and KernelScale: " + meanAccuracies(2));
disp("All parameters: " + meanAccuracies(3));
disp("No optimization (baseline): " + meanAccuracies(4));

disp("Mean of the best accuracies from each experiment: " + meanBestAccuracy);

%% Summary table
summaryTable = array2table(accuracies, 'VariableNames', {'BoxConstraint', ...
                                                        'BoxConstraint_KernelScale', 'All', ...
                                                        'Baseline'});
disp("Summary table of accuracies:");
disp(summaryTable);
%% Save summaryTable in the workspace folder
writetable(summaryTable, 'summaryTable.csv');