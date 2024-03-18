inputimage = [1, 2, 3; 4, 5, 6; 7, 2, 4 ;]
%newimage = [length(inputimage) + 2 , 4  , 5; 3, 5 , 5]
s = size(inputimage)
p = randi([0,0], 5, 5) 
for i = 1 : s(1)
    for j = 1 : s(2)
       p(j+1,i+1) = inputimage(j,i) ;
    end
end
p