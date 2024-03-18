clear all;
I = imread('bikewall.jpg');
U = size(I);
SE = uint8([0, 0 ,0 ; 0, 0 ,0 ; 0, 0, 0]);
SE2 = uint8([1, 1 ,1 ; 1, 1 ,1 ; 1, 1, 1]);

for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        if I(i,j) > 150
            U(i,j) = 1;
        else U(i,j) = 0;
        end
    end
end
imshow(U)
P1 = zeros(size(I,1), size(I,2));
INV = U;
for i = 1 : size(U,1)
    for j = 1 : size(U,2)
        if U(i,j) == 1
            INV(i,j) = 0;
        else INV(i,j) = 1;
        end
    end
end
figure;
imshow(INV);
for i = 2 : size(P1,1) - 1
    for j = 2 : size(P1,2) - 1
        P1(i,j) = sum(union(U(i-1:i+1,j-1:j+1),SE));
        %P1(i,j) = sum(intersect(U(i-1:i+1,j-1:j+1),SE));
    end
end
figure;
P2 = P1;
imshow(P1);
for i = 2 : size(P2,1) - 1
    for j = 2 : size(P2,2) - 1
        P2(i,j) = sum(union(INV(i-1:i+1,j-1:j+1),SE));
        %P1(i,j) = sum(intersect(U(i-1:i+1,j-1:j+1),SE));
    end
end
figure;
imshow(P2);
P2INV = P2;
for i = 1 : size(P2,1)
    for j = 1 : size(P2,2)
        if P2(i,j) == 1
            P2INV(i,j) = 0;
        else P2INV(i,j) = 1;
        end
    end
end
figure;
imshow(P2INV);