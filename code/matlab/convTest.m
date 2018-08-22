
%#img = rand(256,256);
img = imread('cameraman.tif');
img = double(img) / 255
img = img(100:200,100:200)


r = [
					[
						[ 1,3,3,1 ];
						[ 3,9,9,3 ];
						[ 3,9,9,3 ];
						[ 1,3,3,1 ]
					];
					[
						[ 3,9,9,3 ];
						[ 9,27,27,9 ];
						[ 9,27,27,9 ];
						[ 3,9,9,3 ]
					];
					[
						[ 3,9,9,3 ];
						[ 9,27,27,9 ];
						[ 9,27,27,9 ];
						[ 3,9,9,3 ]
					];
					[
						[ 1,3,3,1 ];
						[ 3,9,9,3 ];
						[ 3,9,9,3 ];
						[ 1,3,3,1 ]
					];
				];
r = [
						[ 3,9,9,3 ];
						[ 9,27,27,9 ];
						[ 9,27,27,9 ];
						[ 3,9,9,3 ]
					];           
                
r  = r / sum(sum(r))
%% sec

subplot(3,1,1);
imshow(img);


%k0 = ones(2,2) / (2*2);
%k1 = ones(2,2) / (2*2);
k0 = r;
r
a = [[0,1,0];[1,1,1];[0,1,0]];
a = a / sum(sum(a))
i = ones(2,2) / (2*2)

ra = conv2(r,a)
ra = ra(1:2:end, 1:2:end)
ai = conv2(a,i)

rai0 = conv2(r,ai)
rai1 = conv2(ra,i)



subplot(3,1,2);
res = conv2(img, rai, 'valid');
imshow(res);


subplot(3,1,3);
res = conv2(img, i, 'valid');
imshow(res);


%k1 = ones(4,4) / (4*4)


