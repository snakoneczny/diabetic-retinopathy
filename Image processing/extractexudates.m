%% Extract exudates from one image
tic
fileName = 'E:/Dev/CAD/Diabetic Retinopathy/train/216_right.jpeg';    
% Read image
retinaRGB = imread(fileName);
% Resize image
retinaRGB = resizeretina(retinaRGB, 752, 500);
% Read optic disc mask
opticDiscMask = imread(strrep(fileName, '.jpeg', '_optic_disc_mask.jpeg'));
% Get optic disc mask
opticDiscDilation = 10;
exudatesMask = getexudates(retinaRGB, opticDiscMask, opticDiscDilation);
toc

%% Extract exudates for all images and save them as files
%parpool(4);  % Set workers for paraller computations
tic

% List all images
directory = 'E:/Dev/CAD/Diabetic Retinopathy/train/';
filesLeft = dir(strcat(directory, '*_left.jpeg'));
filesRight = dir(strcat(directory, '*_right.jpeg'));
files = [filesLeft; filesRight];
% For each image
i = 1;
for file = filesRight'
    fileName = strcat(directory, file.name);
    fprintf('Processing image %i / 35126, %s.\n', i, fileName);
    i = i + 1;
    
    if (i > 0)
    % Read image
    retinaRGB = imread(fileName);
    % Resize image
    retinaRGB = resizeretina(retinaRGB, 752, 500);
    % Read optic disc mask
    opticDiscMask = imread(strrep(fileName, '.jpeg', '_optic_disc_mask.jpeg'));
    % Get optic disc mask
    opticDiscDilation = 10;
    exudatesMask = getexudates(retinaRGB, opticDiscMask, opticDiscDilation);
    
    % Save the exudates
    imwrite(exudatesMask, strrep(fileName, '.jpeg', '_exudates.png'))
    
    end
end
toc
%delete(gcp);  % Close threads pool
