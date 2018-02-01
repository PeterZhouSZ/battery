function [ res ] = linIndex(n, x, y, z)

if(size(n,2) == 1)
    res = (x-1)+ n * (y-1) + n*n*(z-1) + 1;
else
    res = (x-1)+ n(1) * (y-1) + n(1)*n(2)*(z-1) + 1;
end

end

