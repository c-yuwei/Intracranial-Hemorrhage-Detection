%***********************
% CNN.m
% Author: Yu-Wei Chang, 17th October, 2019
% CNN.m - Inplace setup of CNN with preprocessed CT dicom files
%
% This script applied the built-in neural network function in MATLAB
% to train a 2D classifier for intracranial hemorrhage using CT dicom file
% provied by open competition RSNA Intracranial Hemorrgage Detection in kaggle.
% 
% revised revision
%----------------------
% 10.18.2019 - added global and local L2 regularization fators and it
%              yielded 84.87% of accuracy on the classification between
%              Epidural and normal.
%***********************

clc;
clear;

% Load image data, specify the folder path of imd datastare
imds = imageDatastore('DataStore', ...
    'IncludeSubfolders',true,'FileExtensions','.dcm','LabelSource','foldernames');
imds.ReadFcn = @customreader;

% Specify traing and validation sets
numTrainFiles = 900;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Define neural network compositions
layers = [
    imageInputLayer([512 512 1]) % specify the input size
    
    convolution2dLayer(8,8,'Padding','same') 
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(8,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    preluLayer(32,'prelu') % Set local L2 regularizatoin factor of layer learnable parameter
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Specify traing options
layers(13) = setL2Factor(layers(13),'Alpha',2); % assign the local L2F to '2'
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0005, ...
    'L2Regularization',0.0005, ... % enlarged the global L2Regularization from 0.0001 to 0.0005
    'MaxEpochs',80, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Training begins...
net = trainNetwork(imdsTrain,layers,options);

% Classify validation set and compute accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)



function data = customreader(filename)
    [data, map] = dicomread(filename);
end
