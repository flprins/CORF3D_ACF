function [scratched_image,  rod_mask] = segmentation(T, R)

v = FlirMovieReader(T);
v.unit = 'temperatureFactory';
while ~isDone(v)
    [frame, metadata] = step(v);
end
frame = imresize(frame, [240 320]);

[r,c] = size(frame);
for i = 1:r
    for j = 1:c
        if frame(i,j) < 30
            frame(i,j) = 0;
%          else
%              im(i,j) = 0;
        end
    end
end


BWsdil = imclose(imclose(frame,strel('line',18,0)),strel('line',18,90));
reg = regionprops(imbinarize(BWsdil));
bw = bwlabel(BWsdil);
[mx,mxind] = max([reg.Area]);
B = double(bw == mxind);

fuse1 = imoverlay(R, ~B, 'black');
fuse2 = imoverlay(B, imbinarize(frame),'black');
fuse2 = rgb2gray(fuse2);

% Smoothen out the edges of mask 320*240
windowSize = 9;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(fuse2), kernel, 'same');
rod_mask = blurryImage > 0.8;

% Scratched RGB Image 320*240
scratched_image = imoverlay(fuse1, rod_mask, 'white');

BB = regionprops(B);
scratched_image = imcrop(scratched_image, BB.BoundingBox);
rod_mask = im2uint8(~rod_mask);

rod_mask = imcrop(~rod_mask, BB.BoundingBox);