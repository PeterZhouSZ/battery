function [] = write_file(evenement,nb_Ellipsoides,R1,R2,fraction,Fraction_volumique, Temps_simulation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% write_file(evenement,nb_Ellipsoides,R1,R2,fraction,Fraction_volumique, Temps_simulation)
%
%% Description :
%  This function creates a text file that contains all important information about
%  the generated ellipsoids (ID, radii, position, orientation, etc.)

%% Implementation

% Extracting the matrix from the cell array
Identite = evenement{1};
Rayon = evenement{3};
Position = evenement{4};
Partenaire = evenement{7};

% Creating and opening the file: ELLIPSOIDS.txt
fid =fopen('ELLIPSOIDS.txt','wt');

% Opening verification
if fid ==-1
    fprintf('Error opening file\n');
else
    fprintf('Creating and opening file "ELLIPSOIDS.txt"\n');
end

% Writing the file header
fprintf(fid,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n');
fprintf(fid,'\t\t\t\t\t\t\t\tGENERATION OF RANDOM ELLIPSOIDS\n\n');
fprintf(fid,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n\n');

% Writing the input and output datas
fprintf(fid,'$$$ USER INPUTS $$$\n\n');
fprintf(fid,'NUMBER OF ELLIPSOIDS: %i\n',nb_Ellipsoides);
fprintf(fid,'ASPECT RATIO a1/a2: %.3f\n',R1);
fprintf(fid,'ASPECT RATIO a1/a3: %.3f\n',R2);
fprintf(fid,'DESIRED VOLUME FRACTION: %.4f\n\n\n',fraction);
fprintf(fid,'$$$ OUTPUT OF THE PROGRAM $$$\n\n');
fprintf(fid,'TIME SIMULATION (s): %.2f\n',Temps_simulation);
fprintf(fid,'REACHED VOLUME FRACTION: %.4f\n\n\n',Fraction_volumique);

% Writing the information about each ellipsoid
fprintf(fid,'$$$ ELLIPSOIDS INFORMATIONS $$$\n\n');
fprintf(fid,'ID\t\tRADIUS\t\t\t\tPOSITION\t\t\tQUATERNIONS\t\t\t\t\t     PARTNER\n\n');

for i=1:size(Identite,1)
    fprintf(fid,'%i\t',Identite(i));
    
    for j = 1:3
        fprintf(fid,'%.3f\t',Rayon(i,j));  
    end
    
    fprintf(fid,'\t');
    
    for j=1:3
        if Position(i,j) >=0
            fprintf(fid,'%.4f\t',Position(i,j));
        else
            fprintf(fid,'%.3f\t',Position(i,j));
        end
    end
    
    fprintf(fid,'\t');
    
    for j=4:7
        if Position(i,j) >=0
            fprintf(fid,'%.4f\t',Position(i,j));
        else
            fprintf(fid,'%.3f\t',Position(i,j));
        end
    end
    
    fprintf(fid,'\t');
    
    L = sum(Partenaire(i,:)~=0);
    for j = 1:L
    fprintf(fid,'%i\t',Partenaire(i,j));
    end
    
    fprintf(fid,'\n');
end
fprintf(fid,'\n\n');

% Writing the end of file
fprintf(fid,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n');
fprintf(fid,'\t\t\t\t\t\t\t\t\tEND OF PROGRAM\n\n');
fprintf(fid,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n\n');

% Closing the file
fclose(fid);

