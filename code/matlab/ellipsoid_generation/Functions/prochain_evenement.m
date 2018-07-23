function [Temps_prochain,check] = prochain_evenement(evenement,nb_Ellipsoides,check)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Temps_prochain] = prochain_evenement(evenement,nb_Ellipsoides)
%
%% Description :
%  Cette fonction permet de calculer le temps du prochain evenement
%  (collision et ou sortie du cube)
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  nb_Sphere: Nombre d'ellipsoides
%
%% Variable de sortie:
%  Temps_prochain     : Ensemble de cellule contenant le temps du prochain
%                       evenement et le ou les ellipsoides impliquées
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Calcul du prochain temps de sortie du cube
[Temps_cube, check] = sortie_cube(evenement,nb_Ellipsoides,check);
% S'il n'y a pas de temps trouvés
if isinf(Temps_cube{1}) == 1
    check{2,2} = ones(nb_Ellipsoides,6);
    [Temps_cube, check] = sortie_cube(evenement,nb_Ellipsoides,check);
end

% Calcul du prochain temps de collision
if evenement{2}(1,1) == 0
    % Le rayon est nul à la première itération. Donc pas de NNL.
    [Temps_collision, check] = interaction(evenement,Temps_cube{1},0,check);
else
    [Temps_collision, check] = interaction(evenement,Temps_cube{1},1,check);
end

% Regroupement des deux temps dans une matrice [1x2]
temps=zeros(1,2);
if isempty(Temps_collision) == 0;
    temps(1) = Temps_collision(1);
else
    temps(1) = inf;
end

if isempty(Temps_cube) == 0
    temps(2) = Temps_cube{1};
else
    temps(2) = inf;
end

% Calcul du temps minimum
[~,indice] = min(temps);

% Creation de l'ensemble de cellule Temps_prochain. 
Temps_prochain=cell(1,4);


if indice == 1
    
    % Collision binaire
    Temps_prochain{1} = Temps_collision(1);
    Temps_prochain{2} = Temps_collision(2);
    Temps_prochain{3} = Temps_collision(3);
    Temps_prochain{4} = 'collision';
    Temps_prochain{5} = Temps_collision(4:6);
    
elseif indice == 2
    
    % Collision entre un ellipsoide et une face
    Temps_prochain{1} = Temps_cube{1};
    Temps_prochain{2} = Temps_cube{2};
    Temps_prochain{3} = Temps_cube{3};
    Temps_prochain{4} = 'cube';
    
end

% % Copie des éléments de Temps_collision dans Temps_prochain
% m=0;
% if abs(temps(1) - tmin) <=1e-15
%     for i=1:size(Temps_collision,1)
%         m=m+1;
%         Temps_prochain{i,1} = Temps_collision(i,1);
%         Temps_prochain{i,2} = Temps_collision(i,2);
%         Temps_prochain{i,3} = Temps_collision(i,3);
%         Temps_prochain{i,4} = 'collision';
%         Temps_prochain{i,5} = Temps_collision(i,4:6);
%
%     end
% end
%
% % Copie des éléments de Temps_secteur dans Temps_prochain
% n=0;
% if abs(temps(2) - tmin) <=1e-15
%     for j=1:size(Temps_cube,1)
%         n=n+1;
%         Temps_prochain{m+j,1} = Temps_cube{j,1};
%         Temps_prochain{m+j,2} = Temps_cube{j,2};
%         Temps_prochain{m+j,3} = Temps_cube{j,3};
%         Temps_prochain{m+j,4} = 'cube';
%     end
% end
%
%
% % Elimination des lignes non utiles de l'ensemble de cellule
% % Temps_prochain
%
% Temps_prochain = Temps_prochain(1:m+n,:);
