function [Fraction] = fraction_volumique(evenement,nb_Ellipsoides)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function fraction_volumique(evenement,nb_Ellipsoides)
%
%% Description : 
%  Cette fonction permet de calculer la fraction volumique des ellipsoides
%  dans le cube
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  nb_Ellipsoides: Nombre d'elipsoides

%% Variable de sortie:
% Fraction : Fraction volumique des ellipsoide         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Extraction des matrices de l'ensemble de cellule.
Rayon = evenement{3};

Volume_ellipsoide = 0;
Volume_cube = 1*1*1;

for i=1:nb_Ellipsoides
    
  Volume_ellipsoide = Volume_ellipsoide + ( (4*pi/3)*(Rayon(i,1)*Rayon(i,2)*Rayon(i,3)) );
  
end

Fraction = Volume_ellipsoide/Volume_cube;
