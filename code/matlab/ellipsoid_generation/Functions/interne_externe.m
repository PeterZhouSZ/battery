function [nouveau_evenement] = interne_externe(evenement,nb_Ellipsoides)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [nouveau_evenement] = apres_cube(evenement,nb_Ellipsoides)
%
%% Description :
%  Cette fonction permet de vérifier si un ellipsoide est complétement sortie
%  ou complétement rentré dans un cube. Dans ce cas, il faut mettre à jour
%  ses partenaires.
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  nb_Ellipsoides: Nombre d'ellipsoides

%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des
%  ellipsoides mis à jour.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1};
Temps = evenement{2};
Rayon = evenement{3};
Position = evenement{4};
Vitesse = evenement{5};
Code = evenement{6};
Partenaire = evenement{7};

for i=1:nb_Ellipsoides
    
    % Extraction des vecteur pertinents à l'ellipsoide i
    Rayon_A = Rayon(i,:);
    Position_A = Position(i,:);
    
    % Calcul des minimums des quadratiques résultants de l'intersection de
    % i avec les 6 face du cube
    [Minimum_1] = minimum_static(Rayon_A,Position_A,1,0);
    [Minimum_2] = minimum_static(Rayon_A,Position_A,1,1);
    [Minimum_3] = minimum_static(Rayon_A,Position_A,2,0);
    [Minimum_4] = minimum_static(Rayon_A,Position_A,2,1);
    [Minimum_5] = minimum_static(Rayon_A,Position_A,3,0);
    [Minimum_6] = minimum_static(Rayon_A,Position_A,3,1);
    
    % Marge
    Marge = 1e-3;
    
    % Conditions d'entrée en x, y et z
    a = (Position(i,1) > 0 && Minimum_1 > Marge) && (Position(i,1) < 1 && Minimum_2 > Marge);
    b = (Position(i,2) > 0 && Minimum_3 > Marge) && (Position(i,2) < 1 && Minimum_4 > Marge);
    c = (Position(i,3) > 0 && Minimum_5 > Marge) && (Position(i,3) < 1 && Minimum_6 > Marge);

    % Conditions de sortie en x, y et z
    d = (Position(i,1) < 0 && Minimum_1 > Marge) || (Position(i,1) > 1 && Minimum_2 > Marge);
    e = (Position(i,2) < 0 && Minimum_3 > Marge) || (Position(i,2) > 1 && Minimum_4 > Marge);
    f = (Position(i,3) < 0 && Minimum_5 > Marge) || (Position(i,3) > 1 && Minimum_6 > Marge);

    % Vérifier si l'ellipsoide est complétement à l'intérieur du cube. On
    % verifie seulement les ellipsoides qui ont deja connue une sortie de cube
    a1 = (a == true && sum(Code(i,[1:2])==1) > 0);
    b1 = (b == true && sum(Code(i,[3:4])==1) > 0);
    c1 = (c == true && sum(Code(i,[5:6])==1) > 0);
    
    if a1 == true || b1 == true  || c1 == true
        
        % Recherche des partenaires
        partenaire_i = Partenaire(i,:);
        
        % Effacer tous les partenaires antérieur de l'ellipsoide
        L = size(Identite,1);
        if sum(partenaire_i) ~= 0
            for k=(nb_Ellipsoides+1):L
                if Identite(k) == partenaire_i(1)
                    
                    Identite = Identite([1:k-1,k+sum(partenaire_i~=0):end]);
                    Temps = Temps([1:k-1,k+sum(partenaire_i~=0):end],:);
                    Rayon = Rayon([1:k-1,k+sum(partenaire_i~=0):end],:);
                    Position = Position([1:k-1,k+sum(partenaire_i~=0):end],:);
                    Vitesse = Vitesse([1:k-1,k+sum(partenaire_i~=0):end],:);
                    Code = Code([1:k-1,k+sum(partenaire_i~=0):end],:);
                    Partenaire = Partenaire([1:k-1,k+sum(partenaire_i~=0):end],:);
                    break
                    
                end
            end
        end
        
        % Mise à jour du code binaire de l'elliposide i
        if a1 == true
            Code(i,:) = [zeros(1,2), Code(i,[3:6])];
        end
        if b1 == true
            Code(i,:) = [ Code(i,[1:2]),zeros(1,2), Code(i,[5:6])];
        end
        if c1 == true
            Code(i,:) = [ Code(i,[1:4]),zeros(1,2)];
        end
        
        % Initialiser la matrice partenaire de la sphère i
        Partenaire(i,:)= zeros(1,7);
        
        % Recréation des nouveaux partenaires dépendement du nouveau code
        % binaire de la sphere.
        if sum(Code(i,:)) ~= 0
            [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, i);
        end
        
        
        % Vérifier si l'ellipsoide est complétement à l'extérieur du cube
    elseif d == true || e == true || f == true
        
        % Recherche des partenaires
        partenaire_e = Partenaire(i,:);
        
        % Identification du code binaire partenaire opposé
        code_oppose = opposer(Code(i,:));
        
        % Recherche du partenaire opposée
        for j = (nb_Ellipsoides+1):size(Identite,1)
            
            % Vérifier que l'ellipsoide j porte le code binaire opposé de l'ellispoide i 
            % et que l'ellipsoide i est un des partenaires de l'ellipsoide j (Double vérification)
            if isequal(Code(j,:), code_oppose) == 1 && sum( Identite(i) == Partenaire(j,:) ) == 1
                
                % Copie des données
                Temps(i,:) = Temps(j,:);
                Rayon(i,:) = Rayon(j,:);
                Position(i,:) = Position(j,:);
                Vitesse(i,:) = Vitesse(j,:);
                
                % Verifier dans quel(s) axe(s) l'ellipsoide est sortiecompletement
                if d == true
                    code_oppose = [zeros(1,2),code_oppose(3:6)];
                end
                if e == true
                    code_oppose = [code_oppose(1:2),zeros(1,2),code_oppose(5:6)];
                end
                if f == true
                    code_oppose = [code_oppose(1:4),zeros(1,2)];
                end
                Code(i,:) = code_oppose;
                
                Partenaire(i,:) = zeros(1,7);
                
                break
            end
        end
        
        % Effacer tous les partenaires antérieur de l'ellipsoide
        L = size(Identite,1);
        if sum(partenaire_e) ~= 0
            for k=(nb_Ellipsoides+1):L
                if Identite(k) == partenaire_e(1)
                    
                    Identite = Identite([1:k-1,k+sum(partenaire_e~=0):end]);
                    Temps = Temps([1:k-1,k+sum(partenaire_e~=0):end],:);
                    Rayon = Rayon([1:k-1,k+sum(partenaire_e~=0):end],:);
                    Position = Position([1:k-1,k+sum(partenaire_e~=0):end],:);
                    Vitesse = Vitesse([1:k-1,k+sum(partenaire_e~=0):end],:);
                    Code = Code([1:k-1,k+sum(partenaire_e~=0):end],:);
                    Partenaire = Partenaire([1:k-1,k+sum(partenaire_e~=0):end],:);
                    break
                    
                end
            end
        end
        
        % Recréation des nouveaux partenaires dépendement du nouveau code
        % binaire de l'ellipsoide
        if sum(Code(i,:)) ~= 0
            [Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, i);
        end
        
        
    end
end


% Insertion des matrices dans l'ensemble de cellule
nouveau_evenement{1} = Identite;
nouveau_evenement{2} = Temps;
nouveau_evenement{3} = Rayon;
nouveau_evenement{4} = Position;
nouveau_evenement{5} = Vitesse;
nouveau_evenement{6} = Code;
nouveau_evenement{7} = Partenaire;

