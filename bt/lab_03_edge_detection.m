clear;
I = imread('bikewall.jpg');
p = imread('bikewall.jpg');
px = imread('bikewall.jpg');
py = imread('bikewall.jpg');
sx = imread('bikewall.jpg');
sy = imread('bikewall.jpg');
rx = size(I);
ry = imread('bikewall.jpg');
conv = imread('bikewall.jpg');


imshow(I)
%BW2 = edge(I, 'Prewitt');
%BW1 = edge(I, 'Canny');
%figure;
%imshowpair(BW1, BW2, 'montage');

kernelx = uint8([1, 1 ,1 ; 0, 0 ,0 ; -1, -1, -1]);
kernely = uint8([1, 0,-1 ; 1, 0, -1;  1,  0, -1]);

for k = 2 : size(p,1) -1
    for l = 2 : size(p,2) -1
        rx = p(k-1,l-1) * kernelx;
        sx(k-1,l-1) = sum(sum(rx));
        %matrisesummen pr pixel
    end
end
for k = 2 : size(p,1) -1
    for l = 2 : size(p,2) -1
        ry = p(k-1,l-1) * kernely;
        sy(k-1,l-1) = sum(sum(ry));
        %matrisesummen pr pixel
    end
end
for k = 2 : size(p,1) -1
    for l = 2 : size(p,2) -1
        conv(k-1,l-1) = (sx(k-1,l-1) + sy(k-1,l-1));
        %c = a * b;
        %sum(sum(c));
        %matrisesummen pr pixel
    end
end
%Maybe convert to binary image?
%imshowpair(I, conv, 'montage');
figure;
imshow(cast(conv,"uint8"));
