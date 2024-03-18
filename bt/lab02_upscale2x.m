dog = imread('dog.jpg');
imshow(dog);
figure;
for i = 1 : size(dog,1) -1
    for j = 1 : size(dog,2) -1
        conv2(2*i-1,2*j-1) = dog(i,j);
        conv2(2*i-1,2*j) = dog(i,j);
        conv2(2*i,2*j-1) = dog(i,j);
        conv2(2*i,2*j) = dog(i,j);
    end
end
imshow(cast(conv2,"uint8"))
figure;
for i = 1 : size(dog,1) -1
    for j = 1 : size(dog,2) -1
        conv(2*i-1,2*j-1) = dog(i,j);
        conv(2*i-1,2*j) = dog(i,j)/2 + dog(i,j+1)/2;
        conv(2*i,2*j-1) = dog(i,j)/2 + dog(i+1,j)/2;
        conv(2*i,2*j) = dog(i,j)/2 + dog(i+1,j+1)/2;
    end
end
imshow(cast(conv,"uint8"))

