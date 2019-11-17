function semanticnucleiHela = semanticsegmentNucleiHelaEM(selectNetwork,Hela) %Complete this part
%--------------------------------------------------------------------------
% Input         Hela                    : an image in Matlab format,it can be 2D/3D, double/uint8
%               selactNetwork           : a deep neural network such as vgg16, ResNet18 or InceptionResNetv2
%               previousSegmentation    : segmentation of slice above/below
%               cannyStdValue           : the value of the Std of the canny edge detection
% Output        nucleiHela              : a binary image with 1 for nuclei, 0 background
%--------------------------------------------------------------------------
% 
% This code performs semantic segmentation for the nuclei of HeLa Cells that have been acquired with Serial Block Face 
% Scanning Electron Microscopy at The Crick Institute by Chris Peddie, Anne Weston, Lucy Collinson and
% provided to the Data Study Group at the Alan Turing Institute by Martin Jones.
%
% The code uses three pre-trained deep neural network that have been fine tuned to perform semantic segmentation to detect the nuclei
% It assumes the following:
%   1 A single cell is of interest and this cell has been cropped from a larger set
%   2 The cell is in the centre of the image
%   3 Although this is a 3D data set, the processing is done on 2D and then
%   post-processed (majority vote) once the whole data stack has been processed.
%   4 Some constants of intensity and size are required, thus this code may only
%   work for the middle region of the cell and not for the top and bottom edges (it
%   was tested for slices 70 to 135
%
%  ------------------------ CITATION ------------------------------------- 
% Part of this work was published in Journal of Imaging:
% Please cite as:
% Cefa Karabag. Martin L. Jones, Christopher J. Peddie, Anne E. Weston, Lucy M. Collinson, and Constantino Carlos Reyes-Aldasoro.  
% Segmentation and Modelling of the Nuclear Envelope of HeLa Cells Imaged with Serial Block Face Scanning Electron Microscopy.  
% Journal of Imaging, 2019; 5(9):75.
%
% Usual disclaimer
%
%--------------------------------------------------------------------------
%
%     Copyright (C) 2019  Cefa Karabag
%
%     This code is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, version 3 of the License.
%
%     The code is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     The GNU General Public License is available at <http://www.gnu.org/licenses/>.
%
%--------------------------------------------------------------------------


%% Download and install Neural Network Toolbox Model for ResNet18, Inception_ResNet_v2 or VGG-16 Network support package. 
% To get started with transfer learning, try choosing one of the faster networks, such as SqueezeNet or GoogLeNet. 
% Then iterate quickly and try out different settings such as data preprocessing steps and training options. 
% Once you have a feeling of which settings work well, try a more accurate network such as Inception-v3 or a ResNet 
% and see if that improves your results. 

%Prompt to enter a number
selectNetwork = input('Enter a number between 1 and 3: ');


switch selectNetwork
    
    case 1
        disp('vgg16')

    case 2
        disp('resnet18')
 
    case 3
         disp('inceptionresnetv2')
end


%% Load Hela Images
    dir0 = 'ROI_1656-6756-329/';
    dir1 =dir(strcat(dir0,'*.tiff'));
    %Hela = imread(strcat(dir0,dir1(118).name));
    %Hela = double(Hela(:,:,1));

    numSlices = size(dir1,1);

%%  This part needs to be done only ONCE and all filtered images have been saved into Deep_Learning_ROI_1656-6756-329
% Convert al Hela images into 2000x2000x3 so they can be fed into vgg16, ResNet18 and InceptionResNetv2 as they expect those dimensions
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

%print('-dtiff','-r400', strcat('Deep_Learning_ROI_1656-6756-329_z000',num2str(k)));

% Use imwrite instead of print as this changes the image dimension
 imwrite(Hela_LPF,['Deep_Learning_ROI_1656-6756-329_z000' num2str(t) '.tiff']);
end

%% % Use imageDatastore to load Hela images. The imageDatastore enables you to efficiently load a large collection of 
% images on disk.

imds = imageDatastore('Deep_Learning_ROI_1656-6756-329/*.tif');

%Display one of the images to test.

I = readimage(imds,118);
I = histeq(I);
imshow(I)

%% Load Hela Pixel-Labeled Images
% Use imageDatastore to load HeLa pixel label image data. 
% A pixelLabelDatastore encapsulates the pixel label data and the label ID to a class name mapping.
% All images were labelled in MATLAB Image Labeler by using 4 different classes.

% Specify these classes.

classes = ["nuclearEnvelope"
           "nucleus" 
           "restOfTheCell" 
           "background"
           ];

%% Supporting function is at the end of the script
cmap = HelaColorMap;
% C = readimage(pxds2,118);

%% 
% Load HeLa Pixel-Labeled Images from Image Labeller 

load('2019_11_14_Net.mat', 'gTruthLabelled') % This was exported from Image Labeler

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
for k0= 1:numSlices
    k1=index2(k0);
C = double(readimage(pxds,k1));
I = readimage(imds,index_px_gt(k1));

% Show overlaid images to check the accuracy 
B = labeloverlay(I,uint8(C),'ColorMap',cmap);
%figure;
imshow(B)
title(strcat(num2str(k1),'-',num2str(k0)))
drawnow;
pause(0.5)

end
pixelLabelColorbar(cmap,classes); % This adds the colorbar in the last figure

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

%% Prepare Training and Test Sets
% SegNet is trained using 60% of the images from the dataset. 
% The rest of the images are used for testing. 
% The following code randomly splits the image and pixel label data into a training and test set.

[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionHelaData(imds,pxds);

%% Create the Network
% Use segnetLayers to create a SegNet network initialized using VGG-16 weights. 
% segnetLayers automatically performs the network surgery needed to transfer the weights from VGG-16 and adds the additional layers required for semantic segmentation.

% imageSize = [360 480 3]; This is for vgg16
% numClasses = numel(classes);
%lgraph = segnetLayers(imageSize,numClasses,'vgg16'); This is for vgg16

%% Create the Network
% helperDeeplabv3PlusResnet18(imageSize, numClasses) creates a
% DeepLab v3+ layer graph object using a pre-trained ResNet-18
% Use the deeplabv3plusLayers function to create a DeepLab v3+ network based on ResNet-18 for InceptionResNetv2. 

% Specify the network image size. This is typically the same as the traing image sizes.
%imageSize = [360 480 3];

% Specify the number of classes.
%numClasses = numel(classes);
% network = 'inceptionresnetv2';

% Create DeepLab v3+ for ResNet18
%lgraph = helperDeeplabv3PlusResnet18(imageSize,numClasses);

% Create DeepLab v3+ for InceptionResNetv2
%lgraph = deeplabv3plusLayers(imageSize,numClasses,network);

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


%% Select Training Options: If you have GPU use the following training options:

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

% For ResNet18 and InceptionResNetv2 use the following training options: 
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
% Data augmentation is used during training to provide more examples to the network because it helps improve the accuracy of the network. 
% Here, random left/right reflection and random X/Y translation of +/- 10 pixels is used for data augmentation.

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

%% Combine the training data and data augmentation selections using pixelLabelImageDatastore. 
% The pixelLabelImageDatastore reads batches of training data, applies data augmentation, 
% and sends the augmented data to the training algorithm.

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

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

[rows_C,cols_C]=size(C);

%% Read all images, resize them, pass them through the net (Deep Learning semantic segmentation algorithm)
% and calculate Jaccard and accuracy for all 300 images

%% Preallocate Jaccard index and accuracy for all 300 slices

JI = zeros(1,10); %numSlices);
Acc= zeros(1,10); %numSlices);

%%
 result = zeros(rows_C,cols_C);
% m = zeros(rows,cols); %numSlices);

for t=1:numSlices
    %disp(t)

m = imread(['Deep_Learning_ROI_1656-6756-329_z000' num2str(t) '.tif']);

IFinal = imresize(m,[360 480]);

CFinal = semanticseg(IFinal, net);
% Display the results.

 result = result +(1*(CFinal=='nuclearEnvelope'));
 result = result +(2*(CFinal=='nucleus'));
 result = result +(3*(CFinal=='restOfTheCell'));
 result = result +(4*(CFinal=='background'));

% Read the corresponding labelled image and resize it

label = imread(['Label_' num2str(t) '.png']);

Label_resized = imresize(label,[360 480]);

% Compare result and labelled image so that you calculate Jaccard and
% accuracy for each slice

currentAccuracy = sum(sum(Label_resized==result))/360/480;
Acc(:,t) = currentAccuracy;

currentJaccard = sum(sum((Label_resized==2).*(result==2)))/sum(sum((Label_resized==2)|(result==2)));
JI(:,t) = currentJaccard;
end

% Although the overall dataset performance is quite high, the class metrics 
% show that underrepresented classes such as nuclearEnvelope is not segmented 
% as well as classes such as nucleus, restOfTheCell, and background. 
% Additional data that includes more samples of the underrepresented 
% classes might help improve the results.

% Plot accuracy and Jaccard index for all three deep neural networks and compare them with the proposed unsupervised algorithm

load('metricsComparison.mat', 'jin');

x = 1:numSlices;

%figure
y1 = Jaccard_vgg16;%(1:231);% This is vgg16 Series Network
%y2 = Jaccard_Our;
y2 = jin; % This was obtained via the proposed algorithm
y3 = Jaccard_ResNet18;%(1:231);% This is ResNet18 DAG Network (Network with Branches)
y4 = Jaccard_Inception_ResNet_v2;%(1:231);% This is Inception_ResNet_v2 DAG Network (Network with Branches)

figure
plot(x,y2,'r',x,y1,'g',x,y3,'b',x,y4,'k','LineWidth',8)

legend('Our algorithm', 'VGG16','ResNet18','InceptionResNetv2','Location','south', 'Fontsize',40)
%legend('boxoff')
%title('Jaccard index comparison between our algorithm and a deep learning algorithm-vgg16','Fontsize',12)
xlabel('Slice number','Fontsize',40)
ylabel('Jaccard index','Fontsize',40)
xlim([25 265])
grid on
ax = gca;
ax.LineWidth = 2;
ax.GridLineStyle = '-';
ax.GridColor = 'k';
ax.GridAlpha = 1;
axis([25 265 0 1])
xticks(25:20:265)
yticks(0:0.2:1)
set(gca,'FontSize',40)

% p(1).LineWidth = 20;
% p(2).LineWidth = 20;


%%


load('metricsComparison.mat', 'Accuracy_Our'); % This was obtained via proposed algorithm

x1 = 1:numSlices; 

figure
y11 = Accuracy_vgg16(1:231);% This is vgg16 Series Network
y22 = Accuracy_Our;
y33 = Accuracy_ResNet18(1:231);% This is ResNet18 DAG Network (Network with Branches)
y44 = Accuracy_Inception_ResNet_v2(1:231);% This is Inception_ResNet_v2 Network (Network with Branches)
figure
plot(x1,y22,'r',x1,y11,'g',x1,y33,'b',x1,y44,'k','LineWidth',8)

legend('Our algorithm', 'VGG16','ResNet18','InceptionResNetv2','Location','south', 'Fontsize',40)
%legend('boxoff')
%title('Jaccard index comparison between our algorithm and a deep learning algorithm-vgg16','Fontsize',12)
xlabel('Slice number','Fontsize',40)
ylabel('Accuracy','Fontsize',40)
xlim([20 220])
grid on
ax = gca;
ax.LineWidth = 2;
ax.GridLineStyle = '-';
ax.GridColor = 'k';
ax.GridAlpha = 1;
%GridStyle.LineWidth = 15;
axis([20 230 0.85 1])
xticks(20:30:230)
yticks(0.85:0.05:1)
set(gca,'FontSize',40)




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
% Define the colormap used by HeLa dataset.
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
% Resize images to [360 480].
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
    I = imresize(I,[360 480]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end
imds = imageDatastore(imageFolder);
end

function pxds = resizeHelaPixelLabels(pxds, labelFolder)
% Resize pixel label data to [360 480].
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
    L = imresize(L,[360 480],'nearest');
    
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



