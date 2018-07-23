function [evenement] = initialiser(nb_Ellipsoides,R1,R2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [evenement] = initialiser(nb_Ellipsoides,R1,R2)
%
%% Description : 
%  A partir d'un nombre d'ellipsoides et des deux rappports de forme, cette
%  fonction permet de g�n�rer 1 ensemble de cellules. Cet ensemble contient
%  l'ensemble des ellipsoides avec leurs caract�ristiques. 
%
%% Variable(s) d'entr�e :
%  nb_Ellipsoides: Nombre d'ellipsoides d�sir�es
%  R1 : rapport de forme a1/a2
%  R2 : rapport de forme a1/a3
%
%% Variable de sortie:
%  evenement: { [Identit�], [Temps], [Rayon Taux de croissance], [Position Quaternion], [Vitesse_lineaire Vitesse_angulaire],
%             , [Code binaire], [partenaire] }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation 

% Taux de croissance maximale des rayons des ellipsodes
a0 = 1;

% Taux de croissance au niveau des trois axes
a1 = 1;
a2 = a1/R1;
a3 = a1/R2;

ai = a0*[a1 a2 a3]/max([a1 a2 a3]);

% Cr�ation de l'ensemble de cellule evenement

evenement{1} = [1:nb_Ellipsoides]';
evenement{2} = zeros(nb_Ellipsoides,2);
evenement{3} = [zeros(nb_Ellipsoides,3),ones(nb_Ellipsoides,1)*ai];
%%
% G�neration d'un axe de rotation al�atoire
r = 2*rand(nb_Ellipsoides,3) - 1;
% G�n�ration d'un angle de rotation al�atoire entre 0 et 2pi
%angle = 2*pi*rand(nb_Ellipsoides,1);
angle = 2*rand(nb_Ellipsoides,1) - 1;
% G�n�ration du quaternion
%Quaternion = [cos(angle/2) repmat((sin(angle/2)./sqrt(r(:,1).^2+r(:,2).^2+r(:,3).^2)),1,3).*r];
Quaternion = [angle repmat((sqrt(1-angle.^2)./sqrt(r(:,1).^2+r(:,2).^2+r(:,3).^2)),1,3).*r];
% Position + Quaternion
evenement{4} = [rand(nb_Ellipsoides,3) Quaternion];
%%
evenement{5} = [2*rand(nb_Ellipsoides,3)-1,0.2*rand(nb_Ellipsoides,3)-0.1];
evenement{6} = zeros(nb_Ellipsoides,6);
evenement{7} = zeros(nb_Ellipsoides,7);
        