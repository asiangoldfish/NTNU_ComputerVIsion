dog = imread('dog.jpg');
imshow(dog);
figure;



p = randi([0,0], 350, 350)
conv = randi([0,0], 5, 5)
inputimage = dog;
%[1, 2, 3; 4, 5, 6; 7, 2, 4 ;]
%newimage = [length(inputimage) + 2 , 4  , 5; 3, 5 , 5]
s = size(inputimage)
kernel = [1, 0 ,1 ; 1, 1 ,1 ; 0 , 1, 0]
for i = 1 : s(1)
    for j = 1 : s(2)
        p(j+1,i+1) = inputimage(j,i) ;
    end
end
p
%concolution
for k = 2 : size(p,1) -1
    for l = 2 : size(p,2) -1
        tmp11 = p (k-1,l-1) * kernel(1,1);
        tmp12 = p (k-1,l) * kernel(1,2);
        tmp13 = p (k-1,l+1) * kernel(1,3);
        tmp21 = p (k,l-1) * kernel(2,1);
        tmp22 = p (k,l) * kernel(2,2);
        tmp23 = p (k,l+1) * kernel(2,3);
        tmp31 = p (k+1,l-1) * kernel(3,1);
        tmp32 = p (k+1,l) * kernel(3,2);
        tmp33 = p (k+1,l+1) * kernel(3,3);
        conv(k,l) = tmp11 + tmp12 + tmp13 + tmp21 + tmp22 + tmp23 + tmp31 + tmp32 + tmp33;
        %c = a * b
        %sum(sum(c))
        %matrisesummen pr pixel
    end
end
%conv;
%imshow(conv)
