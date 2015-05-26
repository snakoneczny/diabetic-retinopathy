%% Extract features from one image
% Load red lesions, exudates and optic disc mask
exudates = imread('E:/Dev/CAD/Diabetic Retinopathy/train/320_right_exudates.jpeg');
opticDisc = imread('E:/Dev/CAD/Diabetic Retinopathy/train/320_right_optic_disc_mask.jpeg');
redLesions = imread('E:/Dropbox/Diabetic Retinopathy/red_lesions_all/320_right_redlesion.png');

% Features 1 - 12 for exudates
featuresExudates = getlesionsfeatures(exudates)
% Features 1 - 12 (13 - 24) for red lesions
featuresRedLesions = getlesionsfeatures(redLesions)
% 12. Optic disc distance from center
opticDistance = getopticdistance(opticDisc)

%% Extract features from all images and save them to file
%parpool(4);  % Set workers for paraller computations
tic

% Initialization
nExamples = 35126;
nFeatures = 33;
features = zeros(nExamples, nFeatures);

% Get all file names 
names = readtable('D:/CAD/diabetic-retinopathy/DRClassification/trainLabels.csv', ...
                  'Delimiter', ',', 'ReadVariableNames', true);
names = names.image;

% For each image
nImages = size(names, 1);
for i = 1 : nImages
    fprintf('Processing image %i / %i, %s.\n', i, nImages, names{i});
    
    % Read images
    exudates = imread(strcat('E:/Dev/CAD/Diabetic Retinopathy/train/', names{i}, '_exudates.jpeg'));
    opticDisc = imread(strcat('E:/Dev/CAD/Diabetic Retinopathy/train/', names{i}, '_optic_disc_mask.jpeg'));
    redLesions = imread(strcat('E:/Dropbox/Diabetic Retinopathy/red_lesions_all/', names{i}, '_redlesion.png'));
    
    % Features 1 - 11 for exudates
    featuresExudates = getlesionsfeatures(exudates, opticDisc);
    % Features 1 - 11 (12 - 22) for red lesions
    featuresRedLesions = getlesionsfeatures(redLesions, opticDisc);
    % 12. Optic disc distance from center
    opticDistance = getopticdistance(opticDisc);
    
    % Save features into matrix
    features(i,:) = [featuresRedLesions, featuresExudates, opticDistance];
end

% Write csv file
csvwrite('D:/CAD/diabetic-retinopathy/DRClassification/features.csv', features);

toc
%delete(gcp);  % Close threads pool
