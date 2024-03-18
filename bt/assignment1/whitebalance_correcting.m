clear;
I = imread('failed_whitebalance.png');
F = imread('rgb_values_failed_wb.png');
imshow(I);
P = size(I);
Q = size(I);
R = size(I);

figure;
rMax = 0;
gMax = 0;
bMax = 0;
for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        if I(i,j,1) > rMax
            rMax = I(i,j,1);
        end
        if I(i,j,2) > gMax
            gMax = I(i,j,2);
        end
        if I(i,j,3) > bMax
            bMax = I(i,j,3);
        end
    end
end

for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        P(i,j,1) = I(i,j,1) * (255/rMax);
        P(i,j,2) = I(i,j,2) * (255/gMax);
        P(i,j,3) = I(i,j,3) * (255/bMax);
        Q(i,j,1) = I(i,j,1) * (105/232);
        Q(i,j,2) = I(i,j,2) * (100/173);
        Q(i,j,3) = I(i,j,3) * (88/78);
    end
end
imshow(cast(P,"uint8"));
figure;
imshow(cast(Q,"uint8"));
rMax = 0;
gMax = 0;
bMax = 0;
rMin = 255;
gMin = 255;
bMin = 255;
for i = 1 : size(Q,1)
    for j = 1 : size(Q,2)
        if Q(i,j,1) > rMax
            rMax = I(i,j,1);
        end
        if Q(i,j,2) > gMax
            gMax = I(i,j,2);
        end
        if Q(i,j,3) > bMax
            bMax = I(i,j,3);
        end
        if Q(i,j,1) < rMin
            rMin = I(i,j,1);
        end
        if Q(i,j,2) < gMin
            gMin = I(i,j,2);
        end
        if Q(i,j,3) < bMin
            bMin = I(i,j,3);
        end
    end
end
%nORMALIZING
for i = 1 : size(I,1)
    for j = 1 : size(I,2)
        R(i,j,1) = Q(i,j,1) * (255/rMax);
        R(i,j,2) = Q(i,j,2) * (255/gMax);
        R(i,j,3) = Q(i,j,3) * (255/bMax);
        %Q(i,j,1) = I(i,j,1) * (105/232);
        %Q(i,j,2) = I(i,j,2) * (100/173);
        %Q(i,j,3) = I(i,j,3) * (88/78);
    end
end
figure;
imshow(cast(R,"uint8"));