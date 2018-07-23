function [Ellipsoide_1, Ellipsoide_2] = collision_rate(evenement,m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Ellipsoide_1, Ellipsoide_2] = collision_rate(evenement)
%
%% Description :
%  Cette fonction permet de vérifier si deux ellipsoides se pénétrent
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  m        : Début de la boucle.
%
%% Variable de sortie:
% Ellipsoide_1 : Identité du premier ellipsoide
% Ellipsoide_2 : Identité du second ellipsoide
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

Ellipsoide_1 = [];
Ellipsoide_2 = [];

Identite = evenement{1};
Rayon = evenement{3};
Position = evenement{4};
Rayon_max = max(evenement{3}(:,1:3),[],2);
Rayon_min = min(evenement{3}(:,1:3),[],2);

%Calcul dea matrices A de tous les ellipsoides à partir de m
A = cell(1,size(Rayon,1));
for i = m:size(Rayon,1)
    
    % Extraction des vecteur pertinents à l'ellipsoide i
    Rayon_A = Rayon(i,:);
    Position_A = Position(i,:);
    % Calcul de la matrice A de l'ellipsoide i
    A{i} = matrice_static(Rayon_A,Position_A);
    
end

stop = 0;
% Marges
Marge1 = 1e-6;
Marge2 = 1e-4;

for i = m:size(Rayon,1)-1
    for j = i+1:size(Rayon,1)
        
        d = sqrt(sum((Position(i,1:3) - Position(j,1:3)).^2));
        
        if d - (Rayon_max(i) + Rayon_max(j)) > Marge1
            continue
            
        elseif d - (Rayon_min(i) + Rayon_min(j)) < -Marge1
            Ellipsoide_1 = Identite(i); 
            Ellipsoide_2 = Identite(j);
            stop = 1;
            break
            
        else
            
            % Calcul des 4 racines
            Racine = eig(A{j},A{i});
            if sum(abs(imag(Racine))) > Marge2  || sum(Racine>0) == 4
                
                Ellipsoide_1 = Identite(i);
                Ellipsoide_2 = Identite(j);
                stop = 1; 
                break
                
            end
            
        end
        
    end
    
    if stop == 1
        break
    end
    
end

