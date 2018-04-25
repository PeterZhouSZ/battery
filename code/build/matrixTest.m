load I_0.dat
load R_0.dat
%%
I_0 = full(spconvert(I_0));
R_0 = full(spconvert(R_0));
R_0 = 8 * R_0;
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

