function [RGB, im4] = segmentation(t, R)
% Convert to gray scale
T = rgb2gray(t);

% Thresholding 320*240
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

% Close and extract maximum area 320*240
BWsdil = imclose(imclose(im,strel('line',18,0)),strel('line',18,90));
% figure, imshow(BWsdil);
reg = regionprops(BWsdil);
bw = bwlabel(BWsdil);
% figure, imagesc(bw);
[mx,mxind] = max([reg.Area]);
B = double(bw == mxind);
B(B == 0) = -1;
% figure,imshow(B);

% Segment complete RGB Cow 320*240
fuse = imoverlay(R,~imbinarize(B), 'black');
%figure,imshow(fuse);

% Obtain mask 320*240
fuse1 = imoverlay(T,~imbinarize(B));
% figure,imshow(fuse1);
fuse2 = rgb2gray(fuse1);  

% Thresholding 320*240 on only cow
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
 
% figure,imshow(im1);
im2 = imbinarize(im1);
% figure,imshow(im2);

% Smoothen out the edges of mask 320*240
windowSize = 9;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(im2), kernel, 'same');
im3 = blurryImage > 0.8; % Rethreshold
%figure,imshow(~im3);

% Scratched RGB Image 320*240
RGB = imoverlay(fuse,~im3, 'white');
% RGB = regionfill(rgb2gray(fuse),~im3);
% figure, imshow(RGB);

BB = regionprops(B);
RGB = imcrop(RGB, BB.BoundingBox);
% RGB = imresize(RGB, [224 224]);
im4 = im2uint8(im3);

im4 = imcrop(~im4, BB.BoundingBox);