function [ exudatesPostprocessed ] = postprocessing( exudates, retina, exudatesMaxSize )
    retina = resizeretina(retina, 752, 500);
    retina = double(retina) / 255;
    retina = sum(retina, 3) ./ 3;

    % Overlay exudates mask on the original image
    % subplot(1, 2, 1), imshow(retina, 'InitialMag', 'fit'), title('original exudates')
    % s = size(retina);
    % green = cat(3, zeros(s), ones(s), zeros(s));
    % hold on
    % h = imshow(green);
    % hold off
    % set(h, 'AlphaData', exudates)

    % Create new exudates mask without large false positives
    measurements = regionprops(exudates, 'Area');
    allAreas = [measurements.Area];
    exudatesValues = find(allAreas < exudatesMaxSize);
    labeledExudates = bwlabel(exudates);
    exudatesPostprocessed = ismember(labeledExudates, exudatesValues);

    % Remove circular shape around retina
    % retinaMask = im2bw(retina, 0.2);
    % retinaMask = imfill(retinaMask, 'holes');
    % se = strel('disk', 16);
    % retinaMask = imerode(retinaMask, se);
    % exudatesPostprocessed = exudatesPostprocessed .* retinaMask;

    % Overlay new exudates mask on the original image
    % subplot(1, 2, 2), imshow(retina, 'InitialMag', 'fit'), title('After postprocessing')
    % s = size(exudates);
    % green = cat(3, zeros(s), ones(s), zeros(s));
    % hold on
    % h = imshow(green);
    % hold off
    % set(h, 'AlphaData', exudatesPostprocessed)
end
