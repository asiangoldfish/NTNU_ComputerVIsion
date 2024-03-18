clear all;
dog = imread('dog.jpg');
imshow(dog);
%figure;
figure;


%p = randi([0,0], 350, 350)
p = dog;
%conv = randi([0,0], 350, 350)
%[1, 2, 3; 4, 5, 6; 7, 2, 4 ;]
%newimage = [length(inputimage) + 2 , 4  , 5; 3, 5 , 5]
s = size(dog);
r = size(dog);

%kernel = [1, 1 ,1 ; 0, 0 ,0 ; -1, -1, -1];
%kernel = [1,0,-1;1,0,-1;1,0,-1];
kernel = uint8([1, 0 ,1 ; 1, 1 ,1 ; 0 , 1, 0])
%kernel = [0 0 0; 0 1 0 ; 0 0 0]
%concolution

for k = 2 : size(p,1) -1
    for l = 2 : size(p,2) -1
        r = p(k-1,l-1) * kernel;
        s(k,l) = sum(sum(r));
        %matrisesummen pr pixel
    end
end
imshow(cast(s,"uint8"))
