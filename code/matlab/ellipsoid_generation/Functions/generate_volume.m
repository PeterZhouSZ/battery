function generate_volume(path, nb_Ellipsoides, fraction, R1, R2, MD_val)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function generate_volume(path, nb_Ellipsoides, fraction, R1, R2, MD_val)
%
%% Description : 
%  Generates a volume of ellipsoids according to the specified parameters
%  and saves it to a file. Also opens a 3D visualization of the volume.
% (Written by Ezra Davis based on code by Alex Doronin)
%% Input variable(s) :
%  path: Folder where the volume will be stored
%  nb_Ellipsoides: number of ellipsoids in the volume
%  fraction: The fraction of the volume that will be filled with the
%  ellipses
%  R1: Ratio between the first and second radii of the
%  ellipsoids.
%  R2: Ratio between the first and third radii of the
%  ellipsoids.
%  MD_val: March-Dollase distributions for the ellipsoids in the volume
%
%% Output variables:
%  None, the output files are written to disk as roughly:
%  "{path}\N_%03d_frac_%.2f_R1_%.2f_R2_%.2f_MD_%.2f_area_%.2f\[1-100].tiff
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clc 

% Computing the final length of the elliposids semi-principle axes
rayon_final = ( (fraction/nb_Ellipsoides)*R1*R2*3/(4*pi) )^(1/3);

tic

% Initialization
fprintf('Initialization of the ellipsoids\n')

[A] = initialiser2(nb_Ellipsoides,R1,R2,MD_val);

nb_Iterations = 0;
Fraction_volumique = 0;

% Creation of matrix temps_elipsoides and ellipsoides_check
check{1,1} = zeros(nb_Ellipsoides,nb_Ellipsoides);
check{1,2} = ones(nb_Ellipsoides,nb_Ellipsoides);
check{2,1} = zeros(nb_Ellipsoides,6);
check{2,2} = ones(nb_Ellipsoides,6);

fprintf('Iterations... Please Wait... !\n')

tic
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
    timeElapsed = toc
    if timeElapsed > 3600
        break
    end
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


% Calculating the surface area of all the (identical and not touching)
% ellipsoids.
Rayon = A{3}; % Radii of the ellipsoids
surface_area = nb_Ellipsoides * ellipsoid_surface_area(Rayon(1,1), Rayon(1,2), Rayon(1,3));
%ratio = sprintf('%.2f',R1/R2);

% Creating and naming the folder:
DATA_DIR = [path, '\', sprintf(...
    'N_%03d_frac_%.2f_R1_%.2f_R2_%.2f_MD_%.2f_area_%.2f', ...
    nb_Ellipsoides, fraction, R1, R2, MD_val, surface_area)];
if ~exist(DATA_DIR,'dir')
    mkdir(DATA_DIR); 
end


for iPosSlice = 1:100
    path = [DATA_DIR,'\',int2str(iPosSlice),'.tiff'];
    slice_data = Vol_data_full(:,:,iPosSlice);
    imwrite(slice_data,path);
end

% Computing the simulation time
Temps_simulation = toc

end