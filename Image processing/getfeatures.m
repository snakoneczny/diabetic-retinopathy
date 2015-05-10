%% Extract features from exudates
% Load exudates and make it logical
exudates = imread('E:/Dev/CAD/Diabetic Retinopathy/sample/16_right_exudates.jpeg');
exudates = im2bw(exudates, 0.1);

% Get measurements
measurements = regionprops(exudates, 'Area', 'Centroid', 'Perimeter')
allCentroids = [measurements.Centroid];
allAreas = [measurements.Area];
allPerimeters = [measurements.Perimeter];
allCircularities = (4 * pi * allAreas) ./ allPerimeters .^ 2;
allCircularities(~isfinite(allCircularities)) = 0;





