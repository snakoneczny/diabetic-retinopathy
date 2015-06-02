function retinaRGB = resizeretina ( retinaRGB, x, y )
% Resize an RGB image of retina
    % subplot(1, 2, 1), imshow(retinaRGB); title('Before resize');
    a = size(retinaRGB, 1);
    b = size(retinaRGB, 2);
    retinaRGB = imresize(retinaRGB, sqrt(x * y / (a * b)));
    % subplot(1, 2, 2), imshow(retinaRGB); title('Resize');
end
