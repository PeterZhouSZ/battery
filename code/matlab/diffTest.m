

d0 = 1.0;
d1 = 0.0001
row = 1;
% m is including ghost points
M = 40;
m = [M, M, M];
h = [1,1,1] ./ (m + [1,1,1]);%(1 / M+2);
invH = [1,1,1] ./ h;
invH2 = [1,1,1] ./ ( h .* h);
n = m(1)*m(2)*m(3);
D = d0 * ones(m);

sn = 40;
minRad = 0.15;
radRange = 0.025;
maxRad = minRad + radRange;
rads = rand([sn,1])*radRange+ minRad;
poss = rand([sn,3]); %* ( 1 - 2*maxRad) + maxRad;
tic
for z=1:m(3)    
    for y=1:m(2)
        for x=1:m(1)             
            v = [(x-1) / (m(1)-1),(y-1) / (m(2) -1), (z-1) / (m(3) -1)];
            
            for k=1:sn
                if ((norm(v - poss(k,:))) < rads(k))
                    D(x,y,z) = d1;
                end
            end
            
        end
    end
end
toc
fprintf('D finished\n')
tic

%D( m(1)/4:end-m(1)/4  ,m(2)/4:end-m(2)/4,m(3)/4:end-m(3)/4) = 0.001;

highConc = 1.0;
lowConc = 0.0;

b = zeros(n,1);

%%
if(M >= 16)
    A = sparse(n,n);
else
    A= zeros(n,n);
end

%diagonal = zeros(n,1);

for z=1:m(3)    
    for y=1:m(2)
        for x=1:m(1)            
            i = linIndex(m,x,y,z);
            Dcur = D(x,y,z);            
            Dxpos = (sampleNeumann(D,x-1,y,z) + Dcur) * 0.5 * invH2(1);
            Dxneg = (sampleNeumann(D,x+1,y,z) + Dcur) * 0.5 * invH2(1);
            Dypos = (sampleNeumann(D,x,y+1,z) + Dcur) * 0.5 * invH2(2);
            Dyneg = (sampleNeumann(D,x,y-1,z) + Dcur) * 0.5 * invH2(2);
            Dzpos = (sampleNeumann(D,x,y,z+1) + Dcur) * 0.5 * invH2(3);
            Dzneg = (sampleNeumann(D,x,y,z-1) + Dcur) * 0.5 * invH2(3);                        
            A(i,i) = -(Dxpos + Dxneg + Dypos + Dyneg + Dzpos + Dzneg);            %correct
            
             if(x < m(1))
                 A(i, linIndex(m,x+1,y,z)) = Dxpos;                 
             else
                 %dirichlet cond
                 %row= i
                 %A(i,i) = A(i,i) - sampleNeumann(D,m+1,y,z) * lowConc;
                 %b(i) = 0 - sampleNeumann(D,m+1,y,z) * lowConc;                 
             end
             
             if(x > 1)
                 A(i, linIndex(m,x-1,y,z)) = Dxneg;
             else
                 %dirichlet cond
                 %A(i, i) = h(1)*h(1);
                 %A(i,i) = A(i,i) - sampleNeumann(D,0,y,z) * highConc;
                 %b(i) = 0 - sampleNeumann(D,0,y,z) * highConc;
             end
             
             if(y < m(2))
                 A(i, linIndex(m,x,y+1,z)) = Dypos;             
             end             
             if(y > 1)
                 A(i, linIndex(m,x,y-1,z)) = Dyneg;             
             end
             if(z < m(3))
                 A(i, linIndex(m,x,y,z+1)) = Dzpos;
             end
             if(z > 1)
                 A(i, linIndex(m,x,y,z-1)) = Dzneg;
             end            
             
             if(y == 1)
                 A(i,i) = A(i,i) + Dyneg;
             end
             
             if(z == 1)
                 A(i,i) = A(i,i) + Dzneg;
             end
             
             if(y == m(2))
                 A(i,i) = A(i,i) + Dypos;
             end
             
             if(z == m(3))
                 A(i,i) = A(i,i) + Dzpos;
             end
                        
             %handle boundary
            %A(i, linIndex(m,x-1,y,z)) = Dxneg;            
        end
        
        %for all boundary nodes
        
        %beg
        begI = linIndex(m,1,y,z);        
        b(begI) = 0 - sampleNeumann(D,0,y,z) * highConc * invH2(1);
        %A(begI,begI) = A(begI,begI) / (h*h);
        %end
        endI= linIndex(m,m(1),y,z);
        b(endI) = 0 - sampleNeumann(D,m+1,y,z) * lowConc * invH2(1);
        %A(endI,endI) = A(endI,endI) / (h*h);
        
    end
end
%%
%A = A * 1/(h*h);
%walls von neumann (y = 1 or m(2), z = 1 or m(2))
%corners -> avg? (chen)
% k = 0;
% sigma = 0;
% for z=1:m(3)    
%     for y=1:m(2)
%         for x=1:m(1)       
%             i = linIndex(m,x,y,z);
%             
%             ybound = 0;
%             zbound = 0;
%             
%             
%             
%             if(y == 1)                   
%                 A(i,i) = A(i,i) - invH(2);
%                 A(i, linIndex(m,x,y+1,z)) = invH(2);
%                 %A(i,i) = A(i,i) + invH(2) * 3/2;                
%                 %A(i, linIndex(m,x,y+1,z)) = A(i, linIndex(m,x,y+1,z)) + invH(2) * (-2);
%                 %A(i, linIndex(m,x,y+2,z)) = A(i, linIndex(m,x,y+2,z)) + invH(2) * (1/2);                          
%                 b(i) = sigma;
%                 ybound = 1;
%             end
%             
%             if(z == 1)
%                 A(i,i) = A(i,i) - invH(3);
%                 A(i, linIndex(m,x,y,z+1)) = invH(3);
%                % A(i,i) = A(i,i) + invH(3) * 3/2;
%                % A(i, linIndex(m,x,y,z+1)) = A(i, linIndex(m,x,y,z+1)) + invH(3) * (-2);
%                % A(i, linIndex(m,x,y,z+2)) = A(i, linIndex(m,x,y,z+2)) + invH(3) * (1/2);                
%                 b(i) = sigma;
%                 zbound = 1;
%             end
%             
%             if(y == m(2))         
%                 A(i,i) = A(i,i) - invH(2);
%                 A(i, linIndex(m,x,y-1,z)) = invH(2);
%                 %A(i,i) = A(i,i) + invH(2) * 3/2;
%                 %A(i, linIndex(m,x,y-1,z)) = A(i, linIndex(m,x,y-1,z)) + invH(2) * (-2);
%                 %A(i, linIndex(m,x,y-2,z)) = A(i, linIndex(m,x,y-2,z)) + invH(2) * (1/2);                
%                 b(i) = sigma;
%                 ybound = 1;
%             end
%             
%             if(z == m(3))     
%                 A(i,i) = A(i,i) - invH(2);
%                 A(i, linIndex(m,x,y,z-1)) = invH(2);
%                 %A(i,i) = A(i,i) + invH(3) * 3/2;
%                 %A(i, linIndex(m,x,y,z-1)) = A(i, linIndex(m,x,y,z-1)) + invH(3) * (-2);
%                 %A(i, linIndex(m,x,y,z-2)) = A(i, linIndex(m,x,y,z-2)) + invH(3) * (1/2);                
%                 b(i) = sigma;
%                 zbound = 1;
%             end
%             
%             
%             
%             
%         end
%     end
% end
%k

toc
fprintf('A finished\n')
%%

%add row/cols for ghost points
%gn = 4*m*m;
%sigma = 0;
%A = [A , zeros(size(A,1),gn)];
%A = [A ; zeros(gn, size(A,2))];
%b = [b ; ones(gn,1) * sigma]
%von neumann cond du/dx = sigma
%b(n:end)



%b

%

%A = [A;zeros(2,n)];
%A(n+1,1) = 1;
%A(n+2,n) = 1;

%b = zeros(n+2,1);
%b(n+1) = 1;
%b = zeros(n,1)

%%
A;
b;

%Ax = b
tic 
x = A \ b;
toc

fprintf('Solver Finished\n')




x = reshape(x,m);
x = permute(x,[3,2,1]);
V = x;

meanx = squeeze(mean(mean(x)));
ax = ((1:m(1)) -1) / (m(1)-1);


clf;
subplot(3,1,1);


plot(ax,meanx)
ylim([0 1.1])
hold on
scatter(ax,meanx,'Ko','filled')
hold off



%%


subplot(3,1,2);
Sx = 1:(m(3) * 10);
Sy = 1:m(2);
Sz = 1:1;
contourslice(V,Sx,Sy,Sz)
view(3)



[qx,qy,qz] = sphere;
hold on;

for k=1:0
    s = surf( ...
        rads(k) * (qx) * m(3) + poss(k,3) * m(3), ...
        rads(k) * (qy) * m(2) + poss(k,2) * m(2), ...
        rads(k) * (qz) * m(1) + poss(k,1) * m(1), ...
        'FaceAlpha',0.1 ...
    );
    s.EdgeColor = 'none';
end
              %if ((norm(v - poss(k,:))) < rads(k))

hold off;

%1/meanx(1)


%%
subplot(3,1,3)
face0 = squeeze(V(:,:,1));
face1 = squeeze(V(:,:,end));
imagesc(face0)

