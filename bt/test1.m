dummy1 = randi([0 9], 5, 5)
scaledMatrix = doubleMatrix(dummy1)


function scaled = doubleMatrix(orig)
    for i = 1 : size(orig, 1)   % rows
        for j = 1 : size(orig, 2)   % columns
            scaled((i + (i-1)),(j + (j-1))) = orig(i,j) %top left
            if i ~= size(orig, 1)
                scaled((i + (i-1) + 1),(j + (j-1))) = (orig(i,j) + orig((i+1),j)) / 2;  % bottom left
            else
                scaled((i + (i-1) + 1),(j + (j-1))) = orig(i, j);    % bottom left
            end
            if j ~= size(orig, 2)
                scaled((i + (i-1)), (j + (j-1) + 1)) = mean([orig(i, j), orig(i, (j+1))]);   % top right
                scaled((i + (i-1) + 1), (j + (j-1) + 1)) = mean([scaled((i + (i-1)), (j + (j-1) + 1)), scaled((i + (i-1) + 1), (j + (j-1)))]);   % bottom right
            else
                scaled((i + (i-1)), (j + (j-1) + 1)) = orig(i, j);   % copy top left to top right
                scaled((i + (i-1) + 1), (j + (j-1) + 1)) = scaled((i + (i-1) + 1), (j + (j-1))); % copy bottom left to bottom right
            end
        end
    end
end