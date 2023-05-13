close all;
clear;


% load the image and pixel label data
imds = imageDatastore(fullfile("daffodilSeg/ImagesRsz256"));


pxDir = fullfile("daffodilSeg/LabelsRsz256");


% classNames = ["background", "daffodil"]
classes = ["flower" "background"];
pixelLabelID = [1 3];
% pixelLabelID = [1 2];
% labelIDs = camvidPixelLabelIDs();

% labelIDs = [1 2];





pxds = pixelLabelDatastore(pxDir,classes,pixelLabelID);

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);







tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')




% [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds, pxds, 'Holdout', 0.2);





% Use the splitEachLabel function to create training and testing datastores




% Split the data into training and validation sets


inputSize = [256 256 3];

imgLayer = imageInputLayer(inputSize)


filterSize = 3;
numFilters = 32;
conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
relu = reluLayer();

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2);

downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    ]

% UPSAMPLE

filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1);

upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    ]

% OUTPUT
numClasses = 2;%should match number of output labels
conv1x1 = convolution2dLayer(1,numClasses);

finalLayers = [
    conv1x1
    softmaxLayer()
    pixelClassificationLayer()
    ]

% STACK together to make full network model
net = [
    imgLayer    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]



DataTrain = combine(imdsTrain,pxdsTrain);

DataVal = combine(imdsVal,pxdsVal);


opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'ValidationData',DataVal,...
    'MaxEpochs',70, ... % how long to train for
    'Plots','training-progress',...
    'MiniBatchSize',10);


% options = trainingOptions('adam', ...
%     'MaxEpochs',70, ...
%     'MiniBatchSize',64, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augmentedValidationSet, ...
%     'ValidationFrequency',5, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

 % pair input images with labels

% trainData = data(trainIdx, :);
% testData = data(testIdx, :);

% DataTrain = combine(imdsTrain,pxdsTrain);
% DataVal = combine(imdsVal,pxdsVal);
% DataTest = combine(imdsTest,pxdsTest);








net = trainNetwork(DataTrain,net,opts);

save('Segmentation.mat', 'net') %filename, variable



pxdsResults = semanticseg(imds,net,"WriteLocation","out");


% metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
% 
% metrics.DataSetMetrics
% metrics.ClassMetrics
% 
% cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
%   classes, Normalization='row-normalized')
% cm.Title = 'Normalized Confusion Matrix (%)'
% 
% imageIoU = metrics.ImageMetrics.MeanIoU;
% figure
% histogram(imageIoU)
% title('Image Mean IoU')
% 
% % Set a threshold for the minimum acceptable IoU
% minIoU = 0.90;
% 
% % Find the indices of the images with an IoU below the threshold
% lowIoUIndices = find(imageIoU < minIoU);
% 
% % Plot the low IoU images
% figure
% for i = 1:numel(lowIoUIndices)
%     % Read in the i-th image and its corresponding ground truth labels
%     image_sample = readimage(imdsTest, lowIoUIndices(i));
%     pxLabel_2 = readimage(pxdsTest, lowIoUIndices(i));
% 
%     % Predict the labels for the image
%     pxLabel_pred = readimage(pxdsResults, lowIoUIndices(i));
% 
%     % Overlay the predicted and ground truth labels on the image
%     overlay_pred = labeloverlay(image_sample, pxLabel_pred,'Transparency', 0.4);
%     overlay_gt = labeloverlay(image_sample, pxLabel_2, 'Transparency', 0.4, 'IncludedLabels', [ "flower"  "background"]);
% 
%     % Display the image and the overlays in two rows of subplots
%     subplot(2,numel(lowIoUIndices),i);
%     imshow(overlay_pred);
%     title(sprintf('Image %d (IoU %.2f)', lowIoUIndices(i), imageIoU(lowIoUIndices(i))));
%     subplot(2,numel(lowIoUIndices),i+numel(lowIoUIndices));
%     imshow(overlay_gt);
%     title(sprintf('Ground Truth %d', lowIoUIndices(i)));
% end




%plotting
I = readimage(imds,1);
figure
imshow(I)


C = readimage(pxds,1);
C(5,5)



%show overlaid groundtruth labels as an example
B = labeloverlay(I,C); %overlay
figure
imshow(B)


overlayOut = labeloverlay(readimage(imds,1),readimage(pxdsResults,1)); %overlay
figure
imshow(overlayOut);
title('overlayOut')

overlayOut = labeloverlay(readimage(imds,2),readimage(pxdsResults,2)); %overlay
figure
imshow(overlayOut);
title('overlayOut2')












function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.70 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = [0 1];

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end





