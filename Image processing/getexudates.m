function exudatesMask = getexudates (retinaRGB, opticDiscMask, artifactsMask, opticDiscDilation)
    %% Get intensity
    % subplot(1, 2, 1), imshow(retinaRGB); title('RGB');
    I = double(retinaRGB) / 255;
    I = sum(I, 3) ./ 3;
    % subplot(1, 2, 2), imshow(I); title('Intensity');

    %% Median filter on intensity channel
    % subplot(1, 2, 1), imshow(I); title('Before median filter');
    I = medfilt2(I);
    % subplot(1, 2, 2), imshow(I); title('Median filter on intensity channel');

    %% Histogram equalization
    % subplot(1, 2, 1), imshow(I); title('Before histogram equalization');
    I = adapthisteq(I);
    % subplot(1, 2, 2), imshow(I); title('Histogram equalization');    

    %% Remove vessels by grayscale closing
    % subplot(1, 2, 1), imshow(I); title('Before grayscale closing');
    se = strel('disk', 8);
    closeI = imclose(I, se);
    % subplot(1, 2, 2), imshow(closeI); title('Grayscale closing');

    %% Local standard deviation of an image
    % subplot(1, 2, 1), imshow(closeI); title('Before standard deviation');
    deviation = stdfilt(closeI, ones(7)); 
    % subplot(1, 2, 2), imshow(deviation, []); title('Standard deviation');

    %% Threshold and dilation
    % subplot(1, 2, 1), imshow(deviation, []); title('Before threshold and dilation');
    level = graythresh(deviation);
    mask = im2bw(deviation, level);
    se = strel('disk', 6);
    mask = imdilate(mask, se);
    % subplot(1, 2, 2), imshow(mask); title('Thresholded and dilated');

    %% Create region of interest
    retinaMask = im2bw(I, 0.2);
    retinaMask = imfill(retinaMask, 'holes');
    se = strel('disk', 16);
    retinaMask = imerode(retinaMask, se);
    % subplot(1, 2, 1), imshow(I), title('Original image');
    % subplot(1, 2, 2), imshow(I, 'InitialMag', 'fit')
    % Make a truecolor all-green image.
    % green = cat(3, zeros(size(I)), ones(size(I)), zeros(size(I)));
    % hold on
    % h = imshow(green);
    % hold off
    % Use our influence map as the AlphaData for the solid green image.
    % set(h, 'AlphaData', retinaMask)

    %% Remove circular shape around retina
    % subplot(1, 2, 1), imshow(mask); title('Before removing circular shape around');
    maskOfCenter = mask .* retinaMask;
    % subplot(1, 2, 2), imshow(maskOfCenter); title('Only region of interest');
    
    %% Flood fill
    % subplot(1, 2, 1), imshow(maskOfCenter); title('Before filling');
    maskFilled = imfill(maskOfCenter, 'holes');
    % subplot(1, 2, 2), imshow(maskFilled); title('Flood filled');

    %% Remove optic disc
    % subplot(1, 2, 1), imshow(maskFilled); title('Before optic disc elimination');
    se = strel('disk', opticDiscDilation);
    opticDiscMask = imdilate(opticDiscMask, se);
    maskOfInterest = uint8(maskFilled) .* imcomplement(opticDiscMask);
    % subplot(1, 2, 2), imshow(maskOfInterest); title('Without optic disc');

    %% Remove artifacts
    % subplot(1, 2, 1), imshow(maskOfInterest); title('Before artifacts elimination');
    se = strel('disk', opticDiscDilation);
    artifactsMask = imdilate(artifactsMask, se);
    maskOfInterest = uint8(maskOfInterest) .* imcomplement(artifactsMask);
    % subplot(1, 2, 2), imshow(maskOfInterest); title('Without artifacts');
    
    %% Overlay mask on the original image
    % subplot(1, 2, 1), imshow(I); title('Before overlay');
    marker = I .* imcomplement(double(maskOfInterest));
    % subplot(1, 2, 2), imshow(marker); title('Overlay');

    %% Reconstruction
    % subplot(1, 2, 1), imshow(marker); title('Before reconstruction');
    reconstructed = imreconstruct(marker, I);
    % subplot(1, 2, 2), imshow(reconstructed); title('Reconstruction');

    %% Threshold on image differences
    diff = I - reconstructed;
    % subplot(1, 2, 1), imshow(diff, []), title('Difference before threshold');
    % level = graythresh(diff)
    level = 0.01;
    exudatesMask = im2bw(diff, level);
    % subplot(1, 2, 2), imshow(exudatesMask), title('Mask');

    %% Overlay exudates mask on the original image
    subplot(1, 2, 1), imshow(retinaRGB), title('Original image');
    subplot(1, 2, 2), imshow(I, 'InitialMag', 'fit')
    % Make a truecolor all-green image.
    green = cat(3, zeros(size(I)), ones(size(I)), zeros(size(I)));
    hold on
    h = imshow(green);
    hold off
    % Use our influence map as the AlphaData for the solid green image.
    set(h, 'AlphaData', exudatesMask)
end
