
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Main Program
%
%  Description :
%  This program takes as inputs the ellipoids number, volume fraction and
%  aspet ratios, and generate the corresponding packing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all
clc
warning off

cd Functions

rng shuffle

% Input Datas (with validation)
nb_Ellipsoides = input('Enter the desired number of ellipsoids: ');
while nb_Ellipsoides <=0 || round(nb_Ellipsoides) ~= nb_Ellipsoides
    fprintf('Invalid entry\n')
    nb_Ellipsoides = input('Enter the desired number of ellipsoids: ');
end

R1 = input('Enter the aspect ratio a1/a2: ');
while R1 <=0
    fprintf('Invalid entry\n')
    R1 = input('Enter the aspect ratio a1/a2: ');
end

R2 = input('Enter the aspect ratio a1/a3: ');
while R2 <=0
    fprintf('Invalid entry\n')
    R2 = input('Enter the aspect ratio a1/a3: ');
end

R3 = input('Enter the March-Dollase r parameter [0-1]: ');
while R3 < 0 ||  R3 > 1
    fprintf('Invalid entry\n')
    R3 = input('Enter the March-Dollase r parameter [0-1]: ');
end

fraction = input('Enter the desired volume fraction: ');
while fraction <=0 || fraction >=1
    fprintf('Invalid entry\n')
    fraction = input('Enter the desired volume fraction: ');
end

% Computing the final length of the elliposids semi-principle axes
rayon_final = ( (fraction/nb_Ellipsoides)*R1*R2*3/(4*pi) )^(1/3);

tic

% Initialization
fprintf('Initialization of the ellipsoids\n')

[A] = initialiser2(nb_Ellipsoides,R1,R2,R3);


nb_Iterations = 0;
Fraction_volumique = 0;

% Creation of matrix temps_elipsoides and ellipsoides_check
check{1,1} = zeros(nb_Ellipsoides,nb_Ellipsoides);
check{1,2} = ones(nb_Ellipsoides,nb_Ellipsoides);
check{2,1} = zeros(nb_Ellipsoides,6);
check{2,2} = ones(nb_Ellipsoides,6);


fprintf('Iterations... Please Wait... !\n')

while (Fraction_volumique - fraction) < 0
    
    nb_Iterations = nb_Iterations + 1;
    
    % Copying the cell array
    B = A;
    
    % Computation of the next event
    [Temps_prochain, check] = prochain_evenement(A,nb_Ellipsoides,check);

    % Update of all ellipsoids
    A = avancer(A,Temps_prochain{1});

    % Verification and re-simulation
    [A,Temps_prochain,check] = verification(A,B,nb_Ellipsoides,Temps_prochain,check);
   
    % Event updating
    A = update_evenement(A, Temps_prochain, nb_Ellipsoides);
   
    % Check if ellipsoids are completely inside or outside the cube
    A = interne_externe(A,nb_Ellipsoides);
   
    % Updating the ellipsoids list to be checked
    check = check_update(check,A,B,Temps_prochain);
    
    % Computation of the volume fraction
    Fraction_volumique = fraction_volumique(A,nb_Ellipsoides)
   
end

% Computing the remaining time 
tps = ( rayon_final - B{3}(1,1) )/B{3}(1,4);
A = avancer(B,tps);

% Final volume fraction
Fraction_volumique = fraction_volumique(A,nb_Ellipsoides)

% Plotting the ellipsoids
fprintf('Plotting the ellipsoids\n')
figure
[Vol_data_full] = tracer2(A);

for iPosSlice = 1:100
    path = ['D:\Battery_code\3D_shape\results\',int2str(iPosSlice),'.tiff'];
    slice_data = Vol_data_full(:,:,iPosSlice);
    imwrite(slice_data,path);
end

%dd
%%

%min_xd = min(xd(:));
%max_xd = max(xd(:));
%step_xd = (max_xd-min_xd)/100;

%min_yd = min(yd(:));
%max_yd = max(yd(:));
%step_yd= (max_yd-min_yd)/100;

%min_zd = min(zd(:));
%max_zd = max(zd(:));
%step_zd = (max_zd-min_zd)/100;

%ti_x = min_xd:step_xd:max_xd; 
%ti_y = min_yd:step_yd:max_yd; 
%ti_z = min_zd:step_zd:max_zd; 


%ti_x = 0:0.01:1;
%ti_y = ti_x;
%ti_z = ti_x;

%v = ones(size(xd));
%[XI,YI,ZI] = meshgrid(ti_x,ti_y,ti_z);
%Vol_data = griddata(xd, yd, zd,v,XI,YI,ZI);
%Vol_data(isnan(Vol_data))=0;

%figure
%plot3(xd, yd, v,'ro')
%hold on
%surf(xd, yd, zd)

%slice_data = Vol_data(:,:,:);
%figure
%imagesc(slice_data)

%h1 = slice(XI,YI,ZI, Vol_data,[],[],min_xd);
%slice_data = get(h1, 'CData');

%[XI,YI,ZI] = meshgrid(ti,ti,ti);
%v = ones(size(xd));
%Vol_data = griddata(xd, yd, zd,v,XI,YI,ZI,'nearest');
%Vol_data(isnan(Vol_data))=0;
%h1 = slice(XI,YI,ZI, Vol_data,[],[],0.9);
%slice_data = get(h1, 'CData');
      
%%
% Computing the simulation time
Temps_simulation = toc

% Writing the text file: "ELLIPSOIDS.txt"
cd ..
write_file(A,nb_Ellipsoides,R1,R2,fraction,Fraction_volumique, Temps_simulation)



