close all;
clear;


% when run, return the path to the folder
rootFolder = fullfile("17flowers/");

categories = {'Bluebell', 'Buttercup', 'ColtsFoot','Cowslip','Crocus','Daffodil','Daisy','Dandelalion','Fritillary','Iris','LilyValley','Pansy','Snowdrop','Sunflower','Tigerlily','Tulip','Windflower'};

% the images in the three categories are now in imds with appropriate
% level.
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
% count images from each label
tbl = countEachLabel(imds);

tbl


[imdsTrain, imdsVal ,imdsTest] = splitEachLabel(imds,0.7, 0.15,0.15, 'randomized');

% Display information about the training, validation, and test sets
disp(imdsTrain);
% % disp(imdsVal);
disp(imdsTest);




% Load the pre-trained ResNet-50 network
% rnd = randperm(numel(imds.Files),9);
% for i = 1:numel(rnd)
% subplot(3,3,i)
% imshow(imread(imds.Files{rnd(i)}))
% label = imds.Labels(rnd(i));
% title(label,"Interpreter","none")
% end


countEachLabel(imdsTrain)
countEachLabel(imdsVal)
countEachLabel(imdsTest)


inputSize = [224 224 3];

augmentedTrainingSet = augmentedImageDatastore(inputSize, ...
    imdsTrain,'ColorPreprocessing','gray2rgb');

augmentedTestingSet = augmentedImageDatastore(inputSize,...
    imdsTest,'ColorPreprocessing','gray2rgb');

augmentedValidationSet = augmentedImageDatastore(inputSize,...
    imdsVal,'ColorPreprocessing','gray2rgb');

% load the ResNet-50 Network

net = resnet50();

% Define the new fully connected layer with 17 output neurons

numClasses = 17;
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc_new')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
]

% Replace the last layer with the new layers

lgraph = layerGraph(net);

% analyzeNetwork(net);
lgraph = removeLayers(lgraph,{'fc1000','fc1000_softmax','ClassificationLayer_fc1000'})
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_new');

% Set the training options
% options = trainingOptions('adam', ...
%     'MiniBatchSize', 32, ...
%     'MaxEpochs', 10, ...
%     'InitialLearnRate', 1e-4, ...
%     'Verbose', true);

options = trainingOptions('adam', ...
    'MaxEpochs',5, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augmentedTrainingSet, lgraph, options);


save("Classification.mat", "net");

% Model Evaluation
load("Classification.mat", "net");
[predictedClasses,predictedScores] = classify(net,augmentedValidationSet);

accuracy = mean(predictedClasses == imdsVal.Labels)

augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

[predictedClasses,predictedScores] = classify(net,augmentedTestingSet);

accuracy = mean(predictedClasses == imdsTest.Labels)





% Compute the confusion matrix
C = confusionmat(imdsTest.Labels, predictedClasses)

% Compute the precision and recall for each class
numClasses = 17
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
for i = 1:numClasses
    truePositives = C(i, i);
    falsePositives = sum(C(:, i)) - truePositives;
    falseNegatives = sum(C(i, :)) - truePositives;
    precision(i) = truePositives / (truePositives + falsePositives);
    recall(i) = truePositives / (truePositives + falseNegatives);
end


% Compute the F1-score for each class
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Compute the macro-averaged F1-score
f1score_macro = mean(f1score);

% Display the F1-scores
disp('F1-score for each class:')
disp(f1score)
disp('Macro-averaged F1-score:')
disp(f1score_macro)




figure;
confusionchart(imdsTest.Labels,predictedClasses,'Normalization',"row-normalized");


% report = classificationReport(imdsTest.Labels, predictedClasses)
% 
% disp(report);




function report = classificationReport(y_true, y_pred)
% Compute a classification report for multi-label classification
% y_true - a categorical array of true labels (samples x 1)
% y_pred - a categorical array of predicted labels (samples x 1)
% report - a structure containing average precision, recall, and F1 score values,
% and accuracy across all labels

% Convert categorical arrays to binary matrices
labels = unique(y_true);
y_true = double(repmat(y_true, 1, numel(labels)) == repmat(labels', numel(y_true), 1));
y_pred = double(repmat(y_pred, 1, numel(labels)) == repmat(labels', numel(y_pred), 1));

% Compute per-label precision, recall, and F1 score
tp = sum(y_true & y_pred, 1);
fp = sum(~y_true & y_pred, 1);
fn = sum(y_true & ~y_pred, 1);
precision = tp ./ (tp + fp);
recall = tp ./ (tp + fn);
f1score = 2 * precision .* recall ./ (precision + recall);

% Compute average precision, recall, and F1 score across all labels
avg_precision = mean(precision);
avg_recall = mean(recall);
avg_f1score = mean(f1score);

% Compute accuracy across all labels
accuracy = mean(mean(y_true == y_pred));

% Create the report structure
report = struct();
report.AveragePrecision = avg_precision;
report.AverageRecall = avg_recall;
report.AverageF1Score = avg_f1score;
report.Accuracy = accuracy;
end

