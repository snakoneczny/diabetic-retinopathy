function [opticDiscMask, artifactsMask] = getopticdiscartifacts (retinaRGB, closingThresholdValue, opticDiscDilationSize, artifactMinSize)
    %% Get intensity
    % subplot(1, 2, 1), imshow(retinaRGB); title('RGB');
    I = double(retinaRGB) / 255;
    I = sum(I, 3) ./ 3;
    % subplot(1, 2, 2), imshow(I); title('Intensity');

    %% Median filter on intensity channel
    % subplot(1, 2, 1), imshow(I); title('Before median filter');
    I = medfilt2(I);
    % subplot(1, 2, 2), imshow(I); title('Median filter on intensity');

    %% Histogram equalization
    % subplot(1, 2, 1), imshow(I); title('Before histogram equalization');
    I = adapthisteq(I);
    % subplot(1, 2, 2), imshow(I); title('Histogram equalization');

    %% Remove vessels by grayscale closing
    % subplot(1, 2, 1), imshow(I); title('Before grayscale closing');
    se = strel('disk', 8);
    closeI = imclose(I, se);
    % subplot(1, 2, 2), imshow(closeI); title('Grayscale closing');

    %% Threshold image to create mask
    % subplot(1, 2, 1), imshow(closeI); title('Before threshold');
    maskFirst = im2bw(closeI, closingThresholdValue);  % Something to do with this hardcoded value
    % subplot(1, 2, 2), imshow(maskFirst); title('Mask');

    %% Overlay mask on the original image
    % subplot(1, 2, 1), imshow(I); title('Before overlay');
    maskFirstRev = imcomplement(maskFirst);
    marker = I .* maskFirstRev;
    % subplot(1, 2, 2), imshow(marker); title('Overlay');

    %% Reconstruction
    % subplot(1, 2, 1), imshow(marker); title('Before reconstruction');
    reconstructed = imreconstruct(marker, I);
    % subplot(1, 2, 2), imshow(reconstructed); title('Reconstruction');

    %% Threshold on image differences and dilate to remove vessels
    diff = I - reconstructed;
    % subplot(1, 2, 1), imshow(diff, []), title('Difference before threshold and dilation');
    level = graythresh(diff);
    opticDiscMask = im2bw(diff, level);
    se = strel('disk', opticDiscDilationSize);
    opticDiscMask = imdilate(opticDiscMask, se);
    % subplot(1, 2, 2), imshow(opticDiscMask), title('Mask');
    
    %% Select optic disc and artifacts from mask
    % subplot(1, 3, 1), imshow(opticDiscMask), title('Before selecting');
    % Get labels and measurement
    labeledDiscMask = bwlabel(opticDiscMask);
    measurements = regionprops(opticDiscMask, 'Area', 'Perimeter');
    % Calculate circularities
    allAreas = [measurements.Area];
    allPerimeters = [measurements.Perimeter];
    allCircularities = (4 * pi * allAreas) ./ allPerimeters .^ 2;
    allCircularities(~isfinite(allCircularities)) = 0;
    % Calculate scores
    allScores = allCircularities .^ 3 .* allAreas;
    
    % Find candidates for optic disc and artifacts
    candidates = find(allAreas > artifactMinSize);
    % No candidates big enough
    if (size(candidates, 2) == 0)
        % Look for smaller one in scores
        [~, opticDiscInd] = max(allScores);
        % Optic disc not found, work around
        if (size(opticDiscInd, 2) == 0)
            opticDiscInd = -1;
        end
        % No artifacts, work around
        artifactsInd = -1;
    % We have candidates
    else
        % Select optic disc as an element with maximum cicularity
        maxCandidateValue = max(allScores(candidates));
        opticDiscInd = find(allScores == maxCandidateValue);
        % Select artifacts as rest of candidates
        artifactsInd = candidates(candidates ~= opticDiscInd);
        % No artifacts, work around
        if (size(artifactsInd) == 0)
            artifactsInd = -1;
        end
    end
    
    % Create image with optic disc
    s = size(labeledDiscMask);
    [r, c] = find(labeledDiscMask == opticDiscInd);
    ind = sub2ind(s, r, c);
    opticDiscMask = zeros(size(opticDiscMask));
    opticDiscMask(ind) = 1;
    
    % Create image with artifacts
    ind = ismember(labeledDiscMask, artifactsInd);
    artifactsMask = zeros(s);
    artifactsMask(ind) = 1;
    
    % Optic disc dilation
    se = strel('disk', 6);
    opticDiscMask = imdilate(opticDiscMask, se);
    
    % Artifacts dilation
    se = strel('disk', 4);
    artifactsMask = imdilate(artifactsMask, se);
    
    % subplot(1, 3, 2), imshow(I .* imcomplement(opticDiscMask)), title('Optic disc extracted');
    % subplot(1, 3, 3), imshow(I .* imcomplement(artifactsMask)), title('Artifacts extracted');
end
