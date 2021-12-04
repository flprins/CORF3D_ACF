
function [temp_map] = temp_segmentation(t)

filename = t;
v = FlirMovieReader(filename);
v.unit = 'temperatureFactory';
while ~isDone(v)
    [frame, metadata] = step(v);
end
frame = imresize(frame, [240 320])

%%%%%%%%Generate mask%%%%%%%%%%%
T = rgb2gray(imread(t));

% Thresholding
[r,c] = size(T);
im = zeros(r,c);
for i = 1:r
    for j = 1:c
        if T(i,j) > 200
            im(i,j) = 1;
%          else
%              im(i,j) = 0;
        end
    end
end
im = bwareaopen(im,60);
im = imclearborder(im,8);
im = imfill(im,'holes');

% Close and extract maximum area
BWsdil = imclose(imclose(im,strel('line',18,0)),strel('line',18,90));
reg = regionprops(BWsdil);
bw = bwlabel(BWsdil);
[mx,mxind] = max([reg.Area]);
B = double(bw == mxind);
B(B == 0) = -1;

% Obtain mask 320*240
fuse1 = imoverlay(T,~imbinarize(B));
fuse2 = rgb2gray(fuse1);

% Thresholding
[r2,c2] = size(fuse2);
im1 = zeros(r2,c2);
 for i = 1:r2
     for j = 1:c2
         if fuse2(i,j) > 200
             im1(i,j) = 1;
%          else
%              im2(i,j) = -1;
         end
     end
 end

im2 = imbinarize(im1);

 % Smoothen out the edges of mask
windowSize = 9;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(im2), kernel, 'same');
im3 = blurryImage > 0.8;

%%%%%%%%%% Inpaint %%%%%%%%%%%%
thermal = regionfill(frame,~im3);
BB = regionprops(B);

%%%%%%%%%%% Crop and threshold%%%%%%%%%%
thermal= imcrop(thermal, BB.BoundingBox);

[r,c] = size(thermal);
im = zeros(r,c);
for i = 1:r
    for j = 1:c
        if thermal(i,j) < 28
            thermal(i,j) = 0;
%          else
%              im(i,j) = 0;
        end
    end
end

temp_map = imresize(thermal, [224 224])
