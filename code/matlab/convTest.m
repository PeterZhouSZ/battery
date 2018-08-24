
%#img = rand(256,256);
img = imread('cameraman.tif');
img = double(img) / 255
img = img(100:200,100:200)




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

%%

ra = conv2(r,a)


%%
ra = conv2(r,a)
ra = ra(1:2:end, 1:2:end)
ai = conv2(a,i)

rai0 = conv2(r,ai)
rai1 = conv2(ra,i)



subplot(3,1,2);
res = conv2(img, rai1, 'valid');
imshow(res);


subplot(3,1,3);
res = conv2(img, i, 'valid');
imshow(res);


%k1 = ones(4,4) / (4*4)




%%

[X4,Y4,Z4] = meshgrid([0,1,2,3],[0,1,2,3],[0,1,2,3]);
X4 = reshape(X4,1,[]);
Y4 = reshape(Y4,1,[]);
Z4 = reshape(Z4,1,[]);

[X3,Y3,Z3] = meshgrid([0,1,2],[0,1,2],[0,1,2]);
X3 = reshape(X3,1,[]);
Y3 = reshape(Y3,1,[]);
Z3 = reshape(Z3,1,[]);

[X2,Y2,Z2] = meshgrid([0,1],[0,1],[0,1]);
X2 = reshape(X2,1,[]);
Y2 = reshape(Y2,1,[]);
Z2 = reshape(Z2,1,[]);



%% 3D convolution

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

a0 = zeros(3,3,3);
a0(2,2,2) = -0.3750;
a0(2,1,2) = 0.0625;
a0(1,2,2) = 0.0625;
a0(2,2,1) = 0.0625;
a0(3,2,2) = 0.0625;
a0(2,2,3) = 0.0625;
a0(2,3,2) = 0.0625;


i0 = ones(2,2,2) / 8;
i0 = reshape([0.0469,0.1406,0.0156,0.0469,0.1406, 0.4219, 0.0469,0.1406], [2,2,2]);

ai = convn(a0,i0)
%a0(a0 == 0.0) = 0.1

%A_0(3767,:)


%%

rviz = abs(reshape(r,1,[]));
rviz = rviz / sum(rviz);

iviz = abs(reshape(i0,1,[]));
iviz = iviz / sum(iviz);

a0viz = abs(reshape(a0,1,[]));
a0viz  = a0viz / sum(a0viz);
a0ind = (a0viz ~= 0.0);

aiviz = abs(reshape(ai,1,[]));
aiviz  = aiviz / sum(aiviz);
aiind = (aiviz ~= 0.0);

colormap jet


subplot(1,3,1)
scatter3(X3(a0ind),Y3(a0ind),Z3(a0ind), abs(a0viz(a0ind)) * 2000,a0viz(a0ind),'filled')
camproj('perspective')
axis vis3d

subplot(1,3,2)
scatter3(X2,Y2,Z2,iviz * 2000,iviz,'filled')
camproj('perspective')
axis vis3d


subplot(1,3,3)
scatter3(X4(aiind),Y4(aiind),Z4(aiind),aiviz(aiind) * 2000,aiviz(aiind) ,'filled')
camproj('perspective')
axis vis3d

