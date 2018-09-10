load ('../build/I_0.dat')
load ('../build/I_1.dat')
load ('../build/I_2.dat')
load ('../build/R_0.dat')
load ('../build/R_1.dat')
load ('../build/R_2.dat')
load ('../build/A_0.dat')
load ('../build/A_1.dat')
load ('../build/A_2.dat')

load ('../build/B_0.txt')

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

%AI_1 = (spconvert(AI_1));
AI_0 = A_0 * I_0;
AI_1 = A_1 * I_1;
AI_2 = A_2 * I_2;
RA_0 = R_0 * A_0;
RA_1 = R_1 * A_1;
RA_2 = R_2 * A_2;
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

%%
%MGGPU vs cpu

clear

%cpu
load ('../build/A_0.dat')
load ('../build/I_0.dat')
load ('../build/R_0.dat')
load ('../build/A_1.dat')

I0CPU = spconvert(I_0);
A0CPU = spconvert(A_0);
R0CPU = spconvert(R_0);
AI0CPU = (A0CPU * I0CPU);
A1CPU = spconvert(A_1);

%gpu
load ('../build/MGGPU_AI_0.dat')
%load ('../build/MGGPU_A_1.dat')
%load ('../build/MGGPU_R_0.dat')
AI0GPU = spconvert(MGGPU_AI_0);
%A1GPU = spconvert(MGGPU_A_1);
%R0GPU = spconvert(MGGPU_R_0);


%RDIFF = full(sum(sum(R0CPU-R0CPU)))
AI0DIFF = full(sum(sum(AI0CPU-AI0GPU)))
%if(RDIFF > 0.0001)
%    error('R0 mismatch')
%end
if(AI0DIFF > 0.0001)
    error('AI0 mismatch')
end


%%

clear

load ('../build/I_1.dat')
load ('../build/A_1.dat')
load ('../build/A_2.dat')
I1CPU = spconvert(I_1);
A1CPU = spconvert(A_1);
A1CPUF = full(A1CPU);
A2CPU = spconvert(A_2);
A2CPUF = full(A2CPU);

load ('../build/MGGPU_I_1.dat')
load ('../build/MGGPU_A_1.dat')
load ('../build/MGGPU_A_2.dat')
I1GPU = spconvert(MGGPU_I_1);
A1GPU = spconvert(MGGPU_A_1);
A1GPUF = full(A1GPU);
A2GPU = spconvert(MGGPU_A_2);
A2GPUF = full(A2GPU);

A1DIFF = full(sum(sum(abs(A1CPU-A1GPU))))
A2DIFF = full(sum(sum(abs(A2CPU-A2GPU))))

clf
subplot(2,2,1)
spy(A1CPU)
subplot(2,2,2)
spy(A1GPU)

subplot(2,2,3)
spy(A2CPU)
subplot(2,2,4)
spy(A2GPU)

%set(gcf, 'Position', get(0, 'Screensize'));
%%
%cusparse sizes

n = 256
lvs = log2(n) - 2;
DBL = 8;
INT = 4;
lvs = 3;

curMem = 0;

for i=0:lvs
    ni = n / (2^i);
    ni3 = ni^3;
    nihalf3 = (ni /2)^3;
    
    if i==0
        aval = 7;
    elseif i==1
        aval = 81;
    elseif i==2
        aval = 64;
    else
        aval = 125;
    end
    
    if i==0
        aival = 20;
    elseif i==1
        aival = 54;
    elseif i==2
        aival = 64;
    else
        aival = 64;
    end
    
    
    ival = 8;    
    rval = 64;    
    
    a = aval * ni3 * DBL + ni3 * INT + ni3 * INT * aval;
    I = ival * ni3 * DBL + ni3 * INT + ni3 * INT * ival;
    R = rval * nihalf3 * DBL + nihalf3 * INT + nihalf3 * INT * rval;
    
    ai = aival * ni3 * DBL + ni3 * INT + ni3 * INT * aival;
    
    totalLevel = (a + I + R + ai) / (1024.0 * 1024.0);
    fprintf('Level %d: %f MB\n',i,totalLevel)
    
end




%%
% plot spy
subplot(1,2,1)
spy(A1CPU)
subplot(1,2,2)
spy(A1GPU)

A1DIFF = full(sum(sum(abs(A1CPU-A1GPU))))

if(A1DIFF > 0.0001)
    error('A0 mismatch')
end

% convert to full
A1GPU = full(A1GPU);
A1CPU = full(A1CPU);






%AI0GPU = full(spconvert(MGGPU_AI_0));
%AI0CPU = full(A0CPU * I0CPU);
%I0CPU = full(I0CPU);
%A0CPU = full(A0CPU);

% Need to see: for 1,1
% -0.3125 * 0.75 + 
% 0.0625 * 0.75 + 
% 0.0625 * 0.5625 + 
% 0.0625 * 0.5625

% Need to see: for 1,2 (+1x)
% -0.3125 * 00000 + 
% 0.0625 * 0.25 + 
% 0.0625 * 0 + 
% 0.0625 * 0

%%
r = 1 + 3 + 3*8 + 3*64  
avec = A0CPU(r,:)
ivec = I0CPU(:,r / 2)
ai0_00 = full(avec) * full(ivec)
AI0CPU(r,r / 2)

%%
load ('../build/I_0.dat')
load ('../build/MGGPU_I_0.dat')
load ('../build/MGGPU_IT_0.dat')

I0TCPU = spconvert(I_0)';
I0TGPU = spconvert(MGGPU_IT_0);

I0TGPUF = full(I0TGPU);
I0TCPUF = full(I0TCPU);

%%
load ('../build/AI_1.dat');
load ('../build/AI_2.dat');
AI_1 = spconvert(AI_1);
AI_2 = spconvert(AI_2);

max(sum(AI_1 ~= 0, 2))
max(sum(AI_2 ~= 0, 2))

%%

x = linspace(1,65356,1000); 
plot(x,x.^3 * 20); 
hold on; 
plot(x,(x/2).^3 * 160); 
hold off;



%%

%%

load ('../build/cmp/x_post_0.txt')
load ('../build/cmp/cpu_x_post_0.txt')

gx = x_post_0;
cx = cpu_x_post_0;

diff = gx-cx;

fprintf('DIFF: %f\n',sum(abs(diff)))

[X,Y,Z] = meshgrid(1:16,1:16,1:16);
X = reshape(X,1,[]);
Y = reshape(Y,1,[]);
Z = reshape(Z,1,[]);


scatter3(X,Y,Z, (abs(diff)*10).^2 + 1)
xlabel('X')
ylabel('Y')
zlabel('Z')




%plot(gx)
%hold on
%plot(cx)

%%
load ('../build/16/MGGPU_A_1.dat')
load ('../build/16/MGGPU_I_0.dat')
load ('../build/16/MGGPU_I_1.dat')
A1_16 = spconvert(MGGPU_A_1);
I0_16 = spconvert(MGGPU_I_0);
I1_16 = spconvert(MGGPU_I_1);

load ('../build/17/MGGPU_A_1.dat')
load ('../build/17/MGGPU_I_0.dat')
load ('../build/17/MGGPU_I_1.dat')
A1_17 = spconvert(MGGPU_A_1);
I0_17 = spconvert(MGGPU_I_0);
I1_17 = spconvert(MGGPU_I_1);

subplot(1,2,1)
spy(A1_16)
%hold on
subplot(1,2,2)
spy(A1_17)
%hold off
%x=get(gca,'children')
%set(x(1),'color','b')
%set(x(2),'color','r')





