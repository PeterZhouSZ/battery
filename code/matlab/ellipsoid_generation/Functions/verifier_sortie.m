function [Valeur] = verifier_sortie(Rayon_A,Position_A,Axe,Face)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Valeur] = verifier_sortie(Rayon_A,Position_A, Vitesse_A,Axe,Face)
%
%% Description :
%  Cette fonction permet de v�rifier si un ellipsoide touche une face du cube
%
%% Variable(s) d'entr�e :
%  Rayon_A:    Rayons de l'ellipsoide A
%  Axe:        L'axe perpendiculaire au plan. Axe = 1 si alpha = x, Axe = 2 si 
%              alpha = y Axe = 3 si alpha = z      
%  Face:       Abscisse de l'intersection entre le plan et l'axe alpha.
%
%% Variable de sortie:
% Valeur : Valeur = 0 s'il n'y a pas de contact et Valeur = 1 s'il y a contact
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation

% Initialisation de la variable Valeur
Valeur = 0;

% Calcul des minimum avec chacune des 6 faces.
Minimum = minimum_static(Rayon_A,Position_A,Axe,Face);

% Marge
Marge = 1e-3;

% V�rification s'il y a contact
if Minimum < Marge
    Valeur = 1;
end

