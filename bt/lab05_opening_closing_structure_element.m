clear all;
InputImage = imread('bikewall.jpg');
%imshow(I);
%figure;
BinaryImage = size(InputImage);
%SE = uint8([0, 0 ,0 ; 0, 0 ,0 ; 0, 0, 0]);
%SE2 = uint8([1, 1 ,1 ; 1, 1 ,1 ; 1, 1, 1]);
%SE = strel("rectangle",[19 8]);
SE = strel("line",4,0);
SE4 = strel("line",4,90);
%SE3 = strel("disk",2);
for i = 1 : size(InputImage,1)
    for j = 1 : size(InputImage,2)
        if InputImage(i,j) > 150
            BinaryImage(i,j) = 1;
        else BinaryImage(i,j) = 0;
        end
    end
end
%imshow(BinaryImage);
%figure;
P1 = imdilate(BinaryImage,SE);
%P2 = P1;
%imshow(P1);
%figure;
P2 = imerode(P1,SE);
%imshow(P2);
P3 = imdilate(P2,SE4);
%figure;
%imshow(P3);
P4 = imerode(P3,SE4);
%figure;
%imshow(P4);
P5 = imerode(P4,SE);
%figure;
%imshow(P5);
montage({BinaryImage,P1 ,P2,P3,P4,P5})