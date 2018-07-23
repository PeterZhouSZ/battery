function [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, ellipsoide)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, ellipsoide)
%
%% Description :
%  Cette fonction permet de v�rifier si une sph�re est compl�tement sortie
%  ou compl�tement rentr� dans un cube. Dans ce cas, il faut mettre � jour
%  ses partenaires.
%
%% Variable(s) d'entr�e :
%  evenement: ensemble de cellule contenant l'ensemble des sph�res
%  sph�re   : Num�ro d'identifiaction de la sph�re dont on veut creer ces
%             partenaires

%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des sph�res
%  ainsi que leur partenaire cr�es
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation
  
% Identification de la sphere
m= ellipsoide;

% Creation de tous les partenaires de l'ellipsoide
% Nombre de partenaire = 1 si l'ellipsoide quitte une face
% Nombre de partenaire = 3 si l'ellipsoide quitte une ar�te
% Nombre de partenaire = 7 si l'ellipsoide quitte un coin
% La fonction permuter(code) sera utilis�.
ensemble_code = permuter(Code(m,:));

% Regroupemement des num�ros d'identifiaction des ellipsoides partenaires
% Les sph�re partenaire sont ajout�es � la fin de l'ensemble de cellule
% nouveau_evenement
L = size(Identite,1);
dernier = Identite(end);
nouveau_partenaire = [m, dernier+1:dernier+length(ensemble_code)];

% Mise � jour des partenaires de la sph�re m
Partenaire(m,:) = [nouveau_partenaire(2:end),zeros(1,7-length(ensemble_code))];

% Allocation d'un nouveau espace pour l'ensemble des matrices
  Identite = [Identite;zeros(length(ensemble_code),1)];
  Temps = [Temps;zeros(length(ensemble_code),2)];
  Rayon = [Rayon;zeros(length(ensemble_code),6)];
  Position = [Position;zeros(length(ensemble_code),7)];
  Vitesse = [Vitesse;zeros(length(ensemble_code),6)];

for q=1:length(ensemble_code)
    
    % Num�ro d'identification
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
   