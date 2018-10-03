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
Arxy = Arx; Arxy(:,2) = Arxy(:,2) .* -1
Ary = A; Ary(:,2) = Ary(:,2) .* -1
Aryz = Ary; Aryz(:,3) = Aryz(:,3) .* -1
Arz = A; Arz(:,3) = Arz(:,3) .* -1
Arzx = Arz; Arzx(:,1) = Arzx(:,1 ) * -1

A = [ A; Arx; Arxy; Arz; Arzx]
b = [b;b;b;b; b]

%%


plotregion(A,b,[],[],[1.0,0.0,0.0]);

axis equal
