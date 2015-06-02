function [ opticDistance ] = getopticdistance( opticDisc )
    center = size(opticDisc) / 2;
    measurements = regionprops(opticDisc, 'Area', 'Centroid');
    if (size(measurements, 1) == 0)
        opticDistance = 0;
    else
        allAreas = [measurements.Area];
        allCentroids = [measurements.Centroid];
        [~, I] = max(allAreas);
        opticDiscCenter = [allCentroids(2 * (I - 1) + 1), ...
                           allCentroids(2 * (I - 1) + 2)];
        opticDistance = sqrt(sum((center - opticDiscCenter) .^ 2));
    end
end
