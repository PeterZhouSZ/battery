load ('../build/I_0.dat')
load ('../build/I_1.dat')
load ('../build/I_2.dat')
load ('../build/R_0.dat')
load ('../build/R_1.dat')
load ('../build/R_2.dat')
load ('../build/A_0.dat')
load ('../build/A_1.dat')
load ('../build/A_2.dat')

load ('../build/AI_1.dat') %A0*I0
%load A_3.dat
%load A_4.dat

%%

%%
I_0 = (spconvert(I_0));
I_1 = (spconvert(I_1));
I_2 = (spconvert(I_2));
R_0 = (spconvert(R_0));
R_1 = (spconvert(R_1));
R_2 = (spconvert(R_2));
A_0 = (spconvert(A_0));
A_1 = (spconvert(A_1));
A_2 = (spconvert(A_2));

AI_1 = (spconvert(AI_1));
%A_3 = (spconvert(A_3));
%A_4 = (spconvert(A_4));
%R_0 = 8 * R_0;
%%
mks0 = max(sum(A_0 ~= 0, 2))
mks1 = max(sum(A_1 ~= 0, 2))
mks2 = max(sum(A_2 ~= 0, 2))

raks = max(sum( (R_0 * A_0) ~= 0, 2))
aiks0 = max(sum( (A_0 * I_0) ~= 0, 2))
aiks1 = max(sum( (A_1 * I_1) ~= 0, 2))

%mks3 = max(sum(A_3 ~= 0, 2))
%mks4 = max(sum(A_4 ~= 0, 2))
%mks5 = max(sum(A_5 ~= 0, 2))

%% convert to list of kernels
%%%% todo



%%
diff = I_0 - R_0';

diff = full(diff);
sum(sum(diff))

%%
r = R_0(5,:);
i = I_0(:,5)';

ie = find(i > 0);
re = find(r > 0);


i(ie)
sum(i(ie))
r(re)
sum(r(re))
ie



% same indices R & I - checked
%%
load A_GPU_0.dat
load A_0.dat
AGPU = full(spconvert(A_GPU_0));
ACPU = full(spconvert(A_0));

