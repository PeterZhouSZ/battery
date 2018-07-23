function [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, ellipsoide)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, ellipsoide)
%
%% Description :
%  Cette fonction permet de vérifier si une sphère est complétement sortie
%  ou complétement rentré dans un cube. Dans ce cas, il faut mettre à jour
%  ses partenaires.
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des sphères
%  sphère   : Numéro d'identifiaction de la sphère dont on veut creer ces
%             partenaires

%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des sphères
%  ainsi que leur partenaire crées
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation
  
% Identification de la sphere
m= ellipsoide;

% Creation de tous les partenaires de l'ellipsoide
% Nombre de partenaire = 1 si l'ellipsoide quitte une face
% Nombre de partenaire = 3 si l'ellipsoide quitte une arête
% Nombre de partenaire = 7 si l'ellipsoide quitte un coin
% La fonction permuter(code) sera utilisé.
ensemble_code = permuter(Code(m,:));

% Regroupemement des numéros d'identifiaction des ellipsoides partenaires
% Les sphère partenaire sont ajoutées à la fin de l'ensemble de cellule
% nouveau_evenement
L = size(Identite,1);
dernier = Identite(end);
nouveau_partenaire = [m, dernier+1:dernier+length(ensemble_code)];

% Mise à jour des partenaires de la sphère m
Partenaire(m,:) = [nouveau_partenaire(2:end),zeros(1,7-length(ensemble_code))];

% Allocation d'un nouveau espace pour l'ensemble des matrices
  Identite = [Identite;zeros(length(ensemble_code),1)];
  Temps = [Temps;zeros(length(ensemble_code),2)];
  Rayon = [Rayon;zeros(length(ensemble_code),6)];
  Position = [Position;zeros(length(ensemble_code),7)];
  Vitesse = [Vitesse;zeros(length(ensemble_code),6)];

for q=1:length(ensemble_code)
    
    % Numéro d'identification
    Identite(L+q) = nouveau_partenaire(q+1);
    
    % Matrice temps
    Temps(L+q,:) = Temps(m,:);
    
    % Matrice Rayon
    Rayon(L+q,:) = Rayon(m,:);
        
    % Matrice vitesse
    Vitesse(L+q,:) = Vitesse(m,:);
    
    % Matrice code binaire
    Code(L+q,:) = ensemble_code{q};
    
    % Matrice partenaire
    Partenaire(L+q,:) = [ nouveau_partenaire(1,[1:q,q+2:end]), zeros(1,7-length(ensemble_code)) ];
    
    % Matrice position
    Position(L+q,1) =  Position(m,1) +  Code(L+q,2) -  Code(m,2);
    Position(L+q,2) =  Position(m,2) +  Code(L+q,4) -  Code(m,4);
    Position(L+q,3) =  Position(m,3) +  Code(L+q,6) -  Code(m,6);
    Position(L+q,4:7) = Position(m,4:7);
    
end
   