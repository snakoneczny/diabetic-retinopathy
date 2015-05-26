function [ features ] = getlesionsfeatures( lesions, opticDisc )
    % Make an image logical
    % lesions = im2bw(lesions, 0.1);
    % opticDisc = im2bw(opticDisc, 0.01);
    
    % Get measurements and check wether lesions exist
    measurements = regionprops(lesions, 'Area', 'Centroid', 'Perimeter');
    if (size(measurements, 1) == 0)
        features = zeros(1, 16);
    else
        
        % 1. Number of lesions
        nLesions = size(measurements, 1);

        % 2. 3. 4. Lesions area total, mean and deviation
        allAreas = [measurements.Area];
        areaTotal = sum(allAreas);
        areaMean = mean(allAreas);
        areaStd = std(allAreas);

        % 5. 6. Circularity mean and deviation
        allPerimeters = [measurements.Perimeter];
        allCircularities = (4 * pi * allAreas) ./ allPerimeters .^ 2;
        allCircularities(~isfinite(allCircularities)) = 0;
        circularityMean = mean(allCircularities);
        circularityStd = std(allCircularities);

        % 7. 8. Distance from center mean and deviation
        center = size(lesions) / 2;
        allDistances = zeros(1, nLesions);
        allCentroids = [measurements.Centroid];
        for i = 1 : nLesions
            allDistances(i) = sqrt((center(1) - allCentroids(2 * (i - 1) + 1)) ^ 2 ...
                                   + (center(2) - allCentroids(2 * (i - 1) + 2)) ^ 2);
        end
        distanceMean = mean(allDistances);
        distanceStd = std(allDistances);

        % 9. 10. Distance from optic disc mean and deviation
        opticDiscMeasurements = regionprops(opticDisc, 'Centroid');
        opticDiscCenter = opticDiscMeasurements.Centroid;
        allOpticDistances = zeros(1, nLesions);
        for i = 1 : nLesions
            allOpticDistances(i) = sqrt((opticDiscCenter(1) - allCentroids(2 * (i - 1) + 1)) ^ 2 ...
                                   + (opticDiscCenter(2) - allCentroids(2 * (i - 1) + 2)) ^ 2);
        end
        opticDistanceMean = mean(allDistances);
        opticDistanceStd = std(allDistances);
        
        % 11. 12. Weighted distance from center mean and deviation
        allWeightedDistances = allDistances .* allAreas;
        weightedDistanceMean = mean(allWeightedDistances);
        weightedDistanceStd = std(allWeightedDistances);

        % 13. 14. Weighted circularities mean and deviation
        allWeightedCircularities = allCircularities .* allAreas;
        weightedCircularityMean = mean(allWeightedCircularities);
        weightedCircularityStd = std(allWeightedCircularities);
        
        % 15. 16. Weighted distance from optic disc mean and deviation
        allWeightedOpticDistances = allOpticDistances .* allAreas;
        weightedOpticDistanceMean = mean(allWeightedOpticDistances);
        weightedOpticDistanceStd = std(allWeightedOpticDistances);

        % Collect all features
        features = [nLesions, areaTotal, areaMean, areaStd, ...
                    circularityMean, circularityStd, ...
                    weightedCircularityMean, weightedCircularityStd, ...
                    distanceMean, distanceStd, ...
                    weightedDistanceMean, weightedDistanceStd, ...
                    opticDistanceMean, opticDistanceStd, ...
                    weightedOpticDistanceMean, weightedOpticDistanceStd];
    end
end
