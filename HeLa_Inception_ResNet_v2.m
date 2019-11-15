%% Download and install Neural Network Toolbox Model for GoogLeNet, Inception-v3 or VGG-16 Network support package. 
% To get started with transfer learning, try choosing one of the faster networks, such as SqueezeNet or GoogLeNet. 
% Then iterate quickly and try out different settings such as data preprocessing steps and training options. 
% Once you have a feeling of which settings work well, try a more accurate network such as Inception-v3 or a ResNet 
% and see if that improves your results. 


switch selectNetwork
    
    case 1
% net = googlenet;

    case 2
% net = vgg16;
 
    case 3
% net = inceptionv3;

    case 4
% net = resnet18;
end


%% In addition, download a pretrained version of SegNet. 
% The pretrained model allows you to run the entire example without having to wait for training to complete.

% pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/segnetVGG16CamVid.mat';
% pretrainedFolder = fullfile(tempdir,'pretrainedSegNet');
% pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat'); 
% if ~exist(pretrainedFolder,'dir')
%     mkdir(pretrainedFolder);
%     disp('Downloading pretrained SegNet (107 MB)...');
%     websave(pretrainedSegNet,pretrainedURL);
% end
%% Setup: This example creates the Deeplab v3+ network with weights initialized from a pre-trained Resnet-18 network. 
% ResNet-18 is an efficient network that is well suited for applications with limited processing resources. 
% Other pretrained networks such as MobileNet v2 or ResNet-50 can also be used depending on application requirements. 

% To get a pretrained Resnet-18, install Deep Learning Toolbox™ Model for Resnet-18 Network. 
% After installation is complete, run the following code to verify that the installation is correct.

q5=inceptionresnetv2;

%% In addition, download a pretrained version of DeepLab v3+. 
% The pretrained model allows you to run the entire example without having to wait for training to complete.

% pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
% pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
% pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
% if ~exist(pretrainedNetwork,'file')
%     mkdir(pretrainedFolder);
%     disp('Downloading pretrained network (58 MB)...');
%     websave(pretrainedNetwork,pretrainedURL);
% end

%% Load Hela Images
% Use imageDatastore to load Hela images. 
% The imageDatastore enables you to efficiently load a large collection of 
% images on disk.


    dir0 = 'ROI_1656-6756-329/';
    dir1 =dir(strcat(dir0,'*.tiff'));% Images
    
   %% 
    %Hela_LPF = zeros(2000);
   
    
%     for idx = 118: 120%numSlices
%          Hela = imread(strcat(dir0,dir1(idx).name));
%          %Hela = double(Hela(:,:,1));
%          Hela = repmat(Hela,[1 1 3]);
%          Hela_LPF = imfilter(Hela,fspecial('Gaussian',7,2)); 
%          imagesc(Hela_LPF); colormap(gray)
%          drawnow
%          pause(0.5)
    %print('-dtiff','-r400', strcat('Deep_Learning_ROI_1656-6756-329_z',num2str(   k)));
         %mkdir('Deep_Learning_ROI_1656-6756-329/');  
        
         %imwrite(Hela_LPF,'Deep_Learning_ROI_1656-6756-329_z000idx.tiff')

    
    %end   

%%  Skip this part as this needs to be done only ONCE and all filtered images have been saved into Deep_Learning_ROI_1656-6756-329
% Convert al Hela images into 2000x2000x3 so they can be fed into vgg16 as it expects those dimensions
% Filter all images and use repmat so all images have dimensions of 2000 x 2000 x 3
for k=1: numSlices
    %figure
    Hela = imread(strcat(dir0,dir1(k).name));
    Hela = repmat(Hela,[1 1 3]);
    Hela_LPF = imfilter(Hela,fspecial('Gaussian',7,2));
  imagesc(Hela_LPF)
    currAxPos = get(gca,'position');
%  if currAxPos(end) == 1
   %set(gca,'position', [0.1300 0.1100 0.7750 0.8150]);axis on
%  else
   set(gca,'position', [0 0 1 1]);axis off
%  end
clear currAxPos

% Print/save Hela_to_Image_labeler(:,:,k) as TIFF images so you can export to Image labeler

print('-dtiff','-r400', strcat('Deep_Learning_ROI_1656-6756-329_z000',num2str(k)));

% Use imwrite instead of print as this changes the image dimension
% imrite(['Deep_Learning_ROI_1656-6756-329_z000' num2str(t) '.tif']);

  %filename = 'Hela_to_Image_labeler';
 %for i = 118:120   
    %print('-dtiff','-r400',filename);
end

%%

%imgDir = fullfile(outputFolder,'images','ROI_1656-6756-329');
%imds = imageDatastore(imgDir);
imds = imageDatastore('Deep_Learning_ROI_1656-6756-329/*.tif');
%Display one of the images.

%%
I = readimage(imds,118);
% I = histeq(I);
imshow(I)

%% Load Hela Pixel-Labeled Images
% Use imageDatastore to load CamVid pixel label image data. 
% A pixelLabelDatastore encapsulates the pixel label data and the label ID 
% to a class name mapping.

% Following the procedure used in original SegNet paper, 
% group the 32 original classes in CamVid to 11 classes. Specify these classes.

classes = ["nuclearEnvelope"
           "nucleus" 
           "restOfTheCell" 
           "background"
           ];

%%
cmap = HelaColorMap;
% C = readimage(pxds2,118);
%% 
% Load HeLa Pixel-Labeled Images from Image Labeller 
pxds = pixelLabelDatastore(gTruthLabelled);

%% Create index to match GT and pxds
for counterImage = 1:300
px_init=6+strfind(pxds.Files{counterImage},'Label_');
px_fin=-1+strfind(pxds.Files{counterImage},'.png');
index_px_gt(counterImage,1) = str2num(pxds.Files{counterImage}(px_init:px_fin));
end
[~,index2]=sort(index_px_gt);

%% %% Resize Hela Data: The images in the data set are 2000 by 2000. 
% To reduce training time and memory usage, resize the images and pixel label images to 720 by 960. 
% resizeCamVidImages and resizeCamVidPixelLabels are supporting functions listed at the end of this example. 

outputFolder = fullfile(tempdir,'Hela');
imageFolder = fullfile(outputFolder,'imagesResized',filesep);
imds = resizeHelaImages(imds,imageFolder);
labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeHelaPixelLabels(pxds,labelFolder);


%%
%C = imresize(double(readimage(pxds,118)),[2000 2000]);
%C = double(readimage(pxds,118));
for k0= 150:160%numSlices
    k1=index2(k0);
C = double(readimage(pxds,k1));
I = readimage(imds,index_px_gt(k1));


B = labeloverlay(I,uint8(C),'ColorMap',cmap);
%figure;
imshow(B)
title(strcat(num2str(k1),'-',num2str(k0)))
drawnow;
pause(0.5)

end
pixelLabelColorbar(cmap,classes);

%% Analyze Dataset Statistics
% To see the distribution of class labels in the dataset use countEachLabel. 
% This function counts the number of pixels by class label.

tbl = countEachLabel(pxds);

% tbl =
% 
%   4×3 table
% 
%           Name           PixelCount    ImagePixelCount
%     _________________    __________    _______________
% 
%     'nuclearEnvelope'    6.9331e+06      9.7169e+08   
%     'nucleus'            9.0477e+07      9.7169e+08   
%     'restOfTheCell'       8.124e+08      1.2207e+09   
%     'background'         3.1092e+08      1.2207e+09   

%% Visualize the pixel counts by class.

frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')


%% Resize Hela Data
% The images in the data set are 2000 by 2000. 
% To reduce training time and memory usage, resize the images and 
% pixel label images to 360 by 480. 

%dir0 = 'ROI_1656-6756-329/'; % As input Hela image coming from raw data
%dir1 =dir(strcat(dir0,'*.tiff'));% Images


% Karabag = zeros(rows,cols,numSlices);
% for k = 118:numSlices
% 
% Karabag(:,:,k) = imresize(double(readimage(pxds2,k)),[360 480]);
% 
% end


%% Use imds and pxds_resized folders as ImageDatastore and PixelLabelDatastore as both are 2000x2000 pixels

% figure ; imagesc(readimage(imds,117))
% figure ; imagesc(Hela_LPF(:,:,117))
% figure ; imagesc(pxds_resized(:,:,117))


%%

% 
  %Hela_resized = zeros(rows,cols,numSlices);
%   Hela_LPF = zeros(rows,cols,numSlices);
% 
% for k=1:numSlices
% 
%     Hela = imread(strcat(dir0,dir1(k).name));
% %     imagesc(Hela)
% %     drawnow
% %     pause(0.5)
%     
%     Hela = double(Hela(:,:,1));
%     Hela_LPF(:,:,k) = imfilter(Hela,fspecial('Gaussian',7,2));
%     %Hela_resized(:,:,k) = imresize(Hela_LPF,[360 480]);
% end


%% Resize filtered images
% confusionchart(known,predicted)

% Hela_resized = zeros(360,480,numSlices);
% 
% for i=1:numSlices
%     
%     Hela_resized(:,:,i) = imresize(Hela_LPF(:,:,i),[360 480]);
% 
% end
%    
%% Resize labelled images 
% Label_resized = zeros(360,480,numSlices);
% 
% for j=1:numSlices
%     
%     Label_resized(:,:,j) = imresize(pxds_resized(:,:,j),[360 480]);
% 
% end

%%
%[training_set, validation_set, testing_set] = splitEachLabel(allImages,.4,.2,.4);

%allImages = imageDatastore('ROI_1656-6756-329/','IncludeSubfolders',true,'LabelSource','foldernames');
%[trainImgs,validationImgs,testImgs] = splitEachLabel(allImages,0.4,0.2,0.4,'randomize');


%allLabels = pixelLabelDatastore(gTruthLabelled);

%[trainLabels,validationLabels,testLabels] = splitEachLabel(allLabels,0.4,0.2,0.4,'randomize');

%[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidData(allImages,allLabels);

%% Prepare Training and Test Sets
% SegNet is trained using 60% of the images from the dataset. 
% The rest of the images are used for testing. 
% The following code randomly splits the image and pixel label data into a training and test set.

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionHelaData(imds,pxds);
% The 60/40 split results in the following number of training and test images:
%[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionHeLaData(imds,pxds);
% numTrainingImages = numel(imdsTrain.Files)
% numTrainingImages = 
%    421
% numTestingImages = numel(imdsTest.Files)
% numTestingImages = 
%    280


%% Create the Network
% Use segnetLayers to create a SegNet network initialized using VGG-16 weights. 
% segnetLayers automatically performs the network surgery needed 
% to transfer the weights from VGG-16 and adds the additional layers required for semantic segmentation.

%vgg16();
%inceptionresnetv2;
% resnet18();
%%
% imageSize = [360 480 3]; This is for vgg16
% numClasses = numel(classes);
%lgraph = segnetLayers(imageSize,numClasses,'vgg16'); This is for vgg16

%% Create the Network
% Use the deeplabv3plusLayers function to create a DeepLab v3+ network based on ResNet-18. 
% Choosing the best network for your application requires empirical analysis and 
% is another level of hyperparameter tuning. 
% You can experiment with different base networks such as ResNet-50 or MobileNet v2, 
% or you can try other semantic segmentation network architectures such as SegNet, 
% fully convolutional networks (FCN), or U-Net.

% Specify the network image size. This is typically the same as the traing image sizes.
%imageSize = [360 480 3];

% Specify the number of classes.
%numClasses = numel(classes);
% network = 'inceptionresnetv2';

% Create DeepLab v3+.
%lgraph = helperDeeplabv3PlusResnet18(imageSize,numClasses);

imageSize = [360 480 3];
numClasses =numel(classes);
network = 'inceptionresnetv2';
lgraph = deeplabv3plusLayers(imageSize,numClasses,network, ...
             'DownsamplingFactor',16);


% The image size is selected based on the size of the images in the dataset. 
% The number of classes is selected based on the classes.

%%
analyzeNetwork(lgraph)



%% Balance Classes Using Class Weighting
% The classes are not balanced. 
% To improve training, you can use class weighting to balance the classes. 
% Use the pixel label counts computed earlier with countEachLayer and 
% calculate the median frequency class weights.

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

%% Specify the class weights using a pixelClassificationLayer.

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);
%% Update the SegNet network with the new pixelClassificationLayer 
% by removing the current pixelClassificationLayer and adding the new layer. 
% The current pixelClassificationLayer is named 'pixelLabels'. 
% Remove it using removeLayers, add the new one using addLayers, 
% and connect the new layer to the rest of the network using connectLayers.

% lgraph = removeLayers(lgraph,'pixelLabels');
% lgraph = addLayers(lgraph, pxLayer);
% lgraph = connectLayers(lgraph,'softmax','labels');


%% Select Training Options
% The optimization algorithm used for training is stochastic gradient descent with momentum (SGDM). 
% Use trainingOptions to specify the hyperparameters used for SGDM.
% nGPUs = gpuDeviceCount;
% 'InitialLearnRate',1e-3 * nGPUs, ...
% 'MiniBatchSize',4 * nGPUs, ...
% 'ExecutionEnvironment', 'multi-gpu'
% options = trainingOptions('sgdm', ...
%     'Momentum',0.9, ...
%     'InitialLearnRate',1e-3, ...
%     'L2Regularization',0.0005, ...
%     'MaxEpochs',100, ...  
%     'MiniBatchSize',4, ...
%     'Shuffle','every-epoch', ...
%     'VerboseFrequency',2, ...
%     'Plots', 'training-progress');

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2,...
    'Plots','training-progress');

% A minibatch size of 4 is used to reduce memory usage while training. 
% You can increase or decrease this value based on the amount of GPU memory you have on your system.

%% Data Augmentation
% Data augmentation is used during training to provide more examples to the network 
% because it helps improve the accuracy of the network. Here, random left/right reflection and 
% random X/Y translation of +/- 10 pixels is used for data augmentation.

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
% imageDataAugmenter supports several other types of data augmentation. 
% Choosing among them requires empirical analysis and is another level of hyperparameter tuning.

%% Combine the training data and data augmentation selections using pixelLabelImageDatastore. 
% The pixelLabelImageDatastore reads batches of training data, applies data augmentation, 
% and sends the augmented data to the training algorithm.

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);
% Start training if the doTraining flag is true. 
% Otherwise, load a pretrained network. 
% Note: Training takes about 5 hours on an NVIDIA™ Titan X and 
% can take even longer depending on your GPU hardware.

%% Network modification
% Modify the network by removing the last three layers. We will replace these layers with new layers for our custom classification.
% layersTransfer = net.Layers(1:end-3);
% Display the output categories.
% categories(imdsTrain.Labels)

% layers = [
%     layersTransfer
%     fullyConnectedLayer(numClasses,'Name', 'fc','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
%     softmaxLayer('Name', 'softmax')
%     classificationLayer('Name', 'classOutput')];

%% Set up a layerGraph and plot it:
% lgraph = layerGraph(layers);
% plot(lgraph)

%% Start Training
[net, info] = trainNetwork(pximds,lgraph,options);

%% Test Network on One Image
% Run the trained network on one test image.

%I = read(imdsTest);

I1 = imread('Deep_Learning_ROI_1656-6756-329_z000118.tif');

I = imresize(I1,[360 480]);

C = semanticseg(I, net);
%Display the results.

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure; imshow(B)
pixelLabelColorbar(cmap, classes);

%% Compare the results in C with the expected ground truth stored in pxdsTest. 
% The green and magenta regions highlight areas where the segmentation results differ from the expected ground truth.

expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);
figure; imshowpair(actual, expected)

%%  Compute Jaccard index, Dice index and other similarity metrics

iou = jaccard(C, expectedResult);
table(classes,iou)

Dice = (2*iou)/(1+iou); 

% The following can also be used for Dice index
% Dice = dice(C, expectedResult);


%% Evaluate Trained Network
% To measure accuracy for multiple test images, run semanticseg on the entire test set.

pxdsResults = semanticseg(imdsTest,net,'MiniBatchSize',4,'WriteLocation',tempdir,'Verbose',false);
% semanticseg returns the results for the test set as a pixelLabelDatastore object. 
% The actual pixel label data for each test image in imdsTest is written to disk in the location specified 
% by the 'WriteLocation' parameter. 

%% Use evaluateSemanticSegmentation to measure semantic segmentation metrics 
% on the test set results. 

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

% evaluateSemanticSegmentation returns various metrics for the entire dataset, for individual classes, 
% and for each test image. 

%% To see the dataset level metrics, inspect metrics.DataSetMetrics .

metrics.DataSetMetrics

% The dataset metrics provide a high-level overview of the network performance. 

%% To see the impact each class has on the overall performance, 
% inspect the per-class metrics using metrics.ClassMetrics.

metrics.ClassMetrics

% Although the overall dataset performance is quite high, the class metrics 
% show that underrepresented classes such as nuclearEnvelope is not segmented 
% as well as classes such as nucleus, restOfTheCell, and background. 
% Additional data that includes more samples of the underrepresented 
% classes might help improve the results.


%% The following functions will be used during training of the deep learning algorithm
% Use the following colorbar to show 3 different classes
% pixelLabelColorbar(Helamap, classes);

 lgraph = helperDeeplabv3PlusResnet18(imageSize, numClasses); 
% creates a DeepLab v3+ layer graph object using a pre-trained ResNet-18 configured
% using the following inputs:
%
%   Inputs
%   ------
%   imageSize    - size of the network input image specified as a vector
%                  [H W] or [H W C], where H and W are the image height and
%                  width, and C is the number of image channels.
%
%   numClasses   - number of classes the network should be configured to
%                  classify.
%
% The output lgraph is a LayerGraph object.







%%

function cmap = HelaColorMap()
% Define the colormap used by CamVid dataset.
cmap = [
    255 0 0   % nuclearEnvelope
    0 255 0       % nucleus
    0 0 255   % restOfTheCell
    0 0 0    % background
    ];
% Normalize between [0 1].
cmap = cmap ./ 255;
end


%% 
function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.
colormap(gca,cmap)
% Add colorbar to current figure.
c = colorbar('peer', gca);
% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);
% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;
% Remove tick mark.
c.TickLength = 0;
end

%%
function imds = resizeHelaImages(imds, imageFolder)
% Resize images to [720 960].
if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end
reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    I = imresize(I,[720 960]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end
imds = imageDatastore(imageFolder);
end

function pxds = resizeHelaPixelLabels(pxds, labelFolder)
% Resize pixel label data to [720 960].
classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
if ~exist(labelFolder,'dir')
    mkdir(labelFolder)
else
    pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
    return; % Skip if images already resized
end
reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    L = imresize(L,[720 960],'nearest');
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(L,[labelFolder filename ext])
end
labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end




%% 
function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionHelaData(imds,pxds)
% Partition HeLa imagesc (data) by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);
% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);
% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);
% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);
% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);
% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

%% Get the definition of TP, TN, FP, and FN so you can compute the following

%    Jaccard(1,:) = TP/(TP+FP+FN);
%    Precision(1,:) = TP/(TP+FP);
%    Recall(1,:) = TP/(TP+FN);
%    Accuracy(1,:) = (TP+TN)/(TP+FN+TN+FP);
%    Dice(1,:) = (2*Jaccard)/(1+Jaccard);
%    dice(A,B) = 2*TP/(2*TP+FP+FN);





