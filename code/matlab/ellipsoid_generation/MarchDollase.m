close all
clear all
clc

theta = 1:1:90;
r_param = 0.2;

nominator = sind(theta);
denominator = (r_param.*r_param.*cosd(theta).*cosd(theta) + (1./r_param).*sind(theta).*sind(theta)).^(3/2);
prob = nominator./denominator;

%prob = prob/max(prob(:));

figure
plot(theta,prob);


A = randpdf(prob, theta, [90, 1])

figure
plot(theta,A)
%imagesc(A);

dd
r=rand
prob=cumsum(prob);
ind=find(r<=prob,1,'first');
x=theta(ind)

%x = cumsum([0 prob(:).'/sum(prob(:))]);
%x(end) = 1e3*eps + x(end);
%[a a] = histc(rand,x);
%F = x(a)


