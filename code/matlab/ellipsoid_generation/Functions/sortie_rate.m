function [Ellipsoide,Face,Cote] = sortie_rate(evenement,nb_Ellipsoides,p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Ellipsoide] = sortie_rate(evenement,nb_Ellipsoides)
%
%% Description :
%  Cette fonction permet de vérifier si un ellipsoide a touché une face du
%  cube sans avoir été détecté
%
%% Variable(s) d'entrée :
%  evenement     : ensemble de cellule contenant l'ensemble des ellipsoidess
%  nb_Ellipsoides: Nombre d'ellipsoides
%  p             : Début de la boucle.
%
%% Variable de sortie:
% Ellipsoide : Identité de l'ellipsoide en question
%  Face      : L'axe perpendiculaire au plan. Face = 1 si alpha = x, Face = 2 si
%              alpha = y Face = 3 si alpha = z
%  Cote      : Abscisse de l'intersection entre le plan et l'axe alpha.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

Ellipsoide = [];
Face = [];
Cote = [];

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1};
Rayon = evenement{3};
Position = evenement{4};
Code = evenement{6};
Rayon_max = max(evenement{3}(:,1:3),[],2);
Rayon_min = min(evenement{3}(:,1:3),[],2);

for i = p:nb_Ellipsoides
    
    % Extraction des vecteur pertinents à l'ellipsoide i
    Rayon_A = Rayon(i,:);
    Position_A = Position(i,:);
    
    % Calcul des minimum avec chacune des 6 faces.
    %Minimum1 = minimum_static(Rayon_A,Position_A,1,0);
    %Minimum2 = minimum_static(Rayon_A,Position_A,1,1);
    %Minimum3 = minimum_static(Rayon_A,Position_A,2,0);
    %Minimum4 = minimum_static(Rayon_A,Position_A,2,1);
    %Minimum5 = minimum_static(Rayon_A,Position_A,3,0);
    %Minimum6 = minimum_static(Rayon_A,Position_A,3,1);
    
    % Marge
    Marge = -1e-6;
    
    % Vérification pour la face 1
    if Code(i,1) == 0 && Position_A(1) - Rayon_max(i) < -Marge
        
        if Position_A(1) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 1;
            Cote = 0;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,1,0);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 1;
                Cote = 0;
                break
            end    
        end
        
    end
    
    % Vérification pour la face 2
    if Code(i,2) == 0 && 1-Position_A(1) - Rayon_max(i) < -Marge
        
        if 1-Position_A(1) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 1;
            Cote = 1;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,1,1);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 1;
                Cote = 1;
                break
            end    
        end
        
    end
    
    % Vérification pour la face 3
    if Code(i,3) == 0 && Position_A(2) - Rayon_max(i) < -Marge
        
        if Position_A(2) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 2;
            Cote = 0;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,2,0);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 2;
                Cote = 0;
                break
            end    
        end
        
    end
    
    % Vérification pour la face 4
    if Code(i,4) == 0 && 1-Position_A(2) - Rayon_max(i) < -Marge
        
        if 1-Position_A(2) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 2;
            Cote = 1;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,2,1);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 2;
                Cote = 1;
                break
            end    
        end
        
    end
    
    % Vérification pour la face 5
    if Code(i,5) == 0 && Position_A(3) - Rayon_max(i) < -Marge
        
        if Position_A(3) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 3;
            Cote = 0;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,3,0);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 3;
                Cote = 0;
                break
            end    
        end
        
    end
    
    % Vérification pour la face 6
    if Code(i,6) == 0 && 1-Position_A(3) - Rayon_max(i) < -Marge
        
        if 1-Position_A(3) - Rayon_min(i) < Marge
            Ellipsoide = Identite(i);
            Face = 3;
            Cote = 1;
            break
        else   
            Minimum = minimum_static(Rayon_A,Position_A,3,1);
            if Minimum < Marge
                Ellipsoide = Identite(i);
                Face = 3;
                Cote = 1;
                break
            end    
        end
        
    end
    
    
end
