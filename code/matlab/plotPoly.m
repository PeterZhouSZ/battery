% Columnar order in jammed LiFePO4 cathodes: ion transport catastrophe and its mitigation
% https://pubs.rsc.org/en/content/articlelanding/2012/cp/c2cp40135e#!divAbstract


close all

AC3=[
    0,1,0;
    1,0,0;
    0.874, 0, 0.486;
    0.804, 0, 0.595;
    0,0,1
   ];
bC3= -[
    0.5;
    9.4;
    10;
    11;
    18
    ]
A = AC3
b = bC3
%%
AC1=[
    0,1,0;
    1,0,0;
    0.874, 0, 0.486;
    0.804, 0, 0.595;
    0,0,1
   ];
bC1= -[
    0.5;
    2.2350;
    2.5;
    2.75;
    4.5
    ]
A = AC1
b = bC1

%%
Arx = A; Arx(:,1) = Arx(:,1) .* -1
Ary = A; Ary(:,2) = Ary(:,2) .* -1
Arz = A; Arz(:,3) = Arz(:,3) .* -1

%Arxy = Arx; Arxy(:,2) = Arxy(:,2) .* -1
%Aryz = Ary; Aryz(:,3) = Aryz(:,3) .* -1
%Arzx = Arz; Arzx(:,1) = Arzx(:,1 ) * -1
Arxz = Arx; Arxz(:,3) = Arxz(:,3) .* -1
Arxy = Arx; Arxy(:,2) = Arxy(:,2) .* -1



A = [ A; Arx; Ary; Arxy ]
b = [b;b ;b; b]

Arz = A; Arz(:,3) = Arz(:,3) .* -1
A = [A;Arz]
b = [b;b]

%%
plotregion(A,b,[],[],[1.0,0.0,0.0]);

hold on
axis equal

%%
addpath('polytopes_2017_10_04_v1.9')


[V,nr,nre]=lcon2vert(A,-b) 

scatter3(V(:,1),V(:,2),V(:,3),'filled')
axis equal
%%
shape = alphaShape(V(:,1),V(:,2),V(:,3));
shape.Alpha = 800.25;
plot(shape)

axis equal

%%
%K = convhulln(V,{'Qt',})
K = convhull(V(:,1),V(:,2),V(:,3),'simplify',true)



%%
nv = int32(size(V,1))
nk = int32(size(K,1))

%conver to zero based
K = int32(K) -1

fid = fopen('particle.txt', 'w');
fprintf(fid,'%d\n',nv);
fclose(fid);

save('particle.txt','V','-ascii','-append')

fid = fopen('particle.txt', 'a');
fprintf(fid,'%d\n',nk);

for i=1:size(K,1)
    fprintf(fid,'%d %d %d %d\n',3, K(i,1),K(i,2),K(i,3));
end

fclose(fid)

%save('particle.txt','K','-ascii','-append')




