

A = load('C:\!\battery\battery\code\build\A.txt');
B = load('C:\!\battery\battery\code\build\b.vec');
load('C:\!\battery\battery\code\python\matrix.txt');
rhs = load('C:\!\battery\battery\code\python\B.txt')';

mult = matrix(2,2)/ A(2,2);
A = (A * mult);
B = B * mult;

d = matrix - A;
db = rhs - B;
spy(d);

r = sum(abs(d'));

N = round(size(A,1)^(1/3));
m = [N,N,N];
for z=1:m(3)    
    for y=1:m(2)
        for x=1:m(1) 
            i = linIndex(m,x,y,z);            
            if(abs(r(i)) > 0.00000001)
            %if(abs(r(i)) < 0.39 && abs(r(i)) > 0.00001)
            %if(abs(r(i)) > 0.2)
                fprintf('row %d (%d %d %d) diff: %f, idx: ',i,x-1,y-1,z-1,r(i));
                
                
                indices = (find(d(i,:)));
                %fprintf('dist: %d ',indices(1) - indices(2)) ;
                for k=1:size(indices,2)
                   j = indices(k);
                   fprintf('%d (x %d), ',j , mod(j-1, 4));                    
                end
                
                fprintf('\n');
                
            end
        end
    end
end

sum(r);

%% diff 0.2 for y|z = 0|3 (y,z boundaries)

%x = load('C:\!\battery\battery\code\build\x.vec')
%xf = [ 0.11766101  0.36663941  0.63335999  0.88233908  0.11083221  0.35776871  0.6422308   0.88916765  0.11083221  0.35776871  0.6422308   0.88916765  0.11766101  0.36663941  0.63335999  0.88233908  0.11083221  0.35776871  0.6422308   0.88916765  0.07873137  0.26506899  0.73493022  0.92126816  0.07873137  0.26506899  0.73493022  0.92126816  0.11083221  0.35776871  0.6422308   0.88916765  0.11083221  0.35776871  0.6422308   0.88916765  0.07873137  0.26506899  0.73493022  0.92126816  0.07873137  0.26506899  0.73493022  0.92126816  0.11083221  0.35776871  0.6422308   0.88916765  0.11766101  0.36663941  0.63335999  0.88233908  0.11083221  0.35776871  0.6422308   0.88916765  0.11083221  0.35776871  0.6422308   0.88916765  0.11766101  0.36663941  0.63335999  0.88233908];
%xf = fliplr(xf)';