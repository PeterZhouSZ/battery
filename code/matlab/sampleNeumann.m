function [ val ] = sampleNeumann( M, x, y, z )

    
    px = x;
    py = y;
    pz = z;
    if(px < 1) px = 1; end
    if(py < 1) py = 1; end
    if(pz < 1) pz = 1; end
    if(px > size(M,1)) px = size(M,1); end
    if(py > size(M,2)) py = size(M,2); end
    if(pz > size(M,3)) pz = size(M,z); end
    
   
    val = M(px, py, pz);
        


end

