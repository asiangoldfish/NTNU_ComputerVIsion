clear all;
InputImage = imread('bikewall.jpg');
%imshow(InputImage);
%figure;
%p1 = medfilt2(InputImage);
p1 = imnoise(InputImage,"salt & pepper",0.02);
p2 = filter2(fspecial('average',3),p1)/255;
p3 = medfilt2(p1);
%imshow(p1);
%figure;
Iblur = imgaussfilt(InputImage,0.5);
Iblur2 = imgaussfilt(InputImage,4);

%imshow(Iblur);
montage({InputImage, p1,p2,p3})