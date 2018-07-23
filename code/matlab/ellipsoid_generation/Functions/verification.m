function [nouveau_evenement,Temps_prochain,check] = verification(evenement,ancien_evenement,nb_Ellipsoides,Temps_prochain,check)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [nouveau_evenement] = verification(evenement,ancien_evenement,nb_Ellipsoides,t0)
%
%% Description :
%  Cette fonction permet de v�rifier si une collision ou une sortie de cube
%  a �t� rat�e. Si oui, on retourne � l'it�ration pr�c�dente et on essayde
%  de recalculer le vrai prochian temps de collision ou de sortie de cube
%  en changeant le point de d�part
%
%% Variable(s) d'entr�e :
%  evenement        : ensemble de cellule contenant l'ensemble des ellipsoides actuel
%  ancien_evenement : ensemble de cellule contenant l'ensemble des ellipsoides � l'it�ration pr�c�dente
%  nb_Ellipsoides   : Nombre d'ellipsoides
%  Tenps_prochain   : Ensemble de cellule contenant le temps du prochain
%                     de evenement et le ou les ellipsoides impliqu�es
%  succes           : succes = 1 si l'op�ration a �t� r�ussi. Sinon, succes = 0;
%% Variable de sortie:
% nouveau_evenement : ensemble de cellule contenant l'ensemble des ellipsoides mis � jour
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation

options= optimset('Display','off','TolFun',1e-15,'TolX',1e-15,'Maxiter',inf,'MaxFunevals',inf);

% Extraction du temps erron�s
t0 = Temps_prochain{1};

% Extraction des matrices importantes
Identite = ancien_evenement{1};
Rayon = ancien_evenement{3};
Position = ancien_evenement{4};
Vitesse = ancien_evenement{5};

% Verification s'il y a penetration ou sortie de cube
m=1;
p=1;
[Ellipsoide_1, Ellipsoide_2] = collision_rate(evenement,m);
[Ellipsoide,Face,Cote] = sortie_rate(evenement,nb_Ellipsoides,p);

% Initialisation des variables
Ellipsoide_ancien = [];
Ellipsoide_1_ancien = [];
Ellipsoide_2_ancien = [];
essai = -1;
output = 1;

%%

while (isempty(Ellipsoide_1) == 0 || isempty(Ellipsoide) == 0)
    
    essai = essai + 1;
    if essai > 100
        output = 2;
        essai = 0;
    end
    
    % Enregistrement des anciennes valeurs
    Ellipsoide_1_ancien = Ellipsoide_1;
    Ellipsoide_2_ancien = Ellipsoide_2;
    Ellipsoide_ancien = Ellipsoide;
    Face_ancien = Face;
    Cote_ancien = Cote;
    
    % Cas o� une collision a �t� rat�e
    if isempty(Ellipsoide_1) == 0
        
        % Recherche des ellipsoides concern�es
        m = find(Identite == Ellipsoide_1);
        n =  find(Identite == Ellipsoide_2);
        
        % Extraction des matrices importantes
        Rayon_A = Rayon(m,:);
        Rayon_B = Rayon(n,:);
        Position_A = Position(m,:);
        Position_B = Position(n,:);
        Vitesse_A = Vitesse(m,:);
        Vitesse_B = Vitesse(n,:);
        
        % Calcul du nouveaux temps de collision avec un diff�rent point de d�part
        if essai < 100
            dt = fzero(@(dt) polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B,output),essai*sqrt(t0)/100,options);
        elseif essai == 100
            dt = fzero(@(dt) polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B,output),1e-15,options);
        end
        

        % Remise � jour du temps de collision dans la matrice
        % temps_ellipsoides_1 de l'ensemble de cellule check
        check{1,1}(m,n) = dt^2;
        check{1,1}(n,m) = dt^2;
        
        % Cas o� une sortie de cube a �t� rat�e
    elseif isempty(Ellipsoide) == 0
        
        % Recherche de l'ellipsoide concern�e
        p = find(Identite == Ellipsoide);
        
        % Extraction des matrices importantes
        Rayon_A = Rayon(p,:);
        Position_A = Position(p,:);
        Vitesse_A = Vitesse(p,:);
        
        % Calcul du nouveaux temps de collision avec un diff�rent point de d�part
        if essai < 100
            dt = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,Face,Cote),essai*sqrt(t0)/100,options);
        elseif essai == 100
            dt = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,Face,Cote),1e-15,options);
        end

        % Remise � jour du temps de collision dans la matrice
        % temps_ellipsoides_2 de l'ensemble de cellule check
        check{2,1}(p,2*Face_ancien+Cote_ancien-1) = dt^2;
        
    end

    % On fait avancer les ellipsoides seulement si le temps trouv� est
    % inf�rieur au temps initial erronn� (sinon �a sert � rien d'avancer)
    if dt^2 < t0
        % Mise � jour de tous les ellipsoides
        evenement = avancer(ancien_evenement,dt^2); 
    end
    
    % Verification s'il y a penetration ou sortie de cube (� partir de o�
    % on s'est arete avant: m ou p)
    [Ellipsoide_1, Ellipsoide_2] = collision_rate(evenement,m);
    [Ellipsoide,Face,Cote] = sortie_rate(evenement,nb_Ellipsoides,p);

    % Remise du compteur � 0 si d'autres ellipsoids overlap 
     if (Ellipsoide_1 ~= Ellipsoide_1_ancien & Ellipsoide_2 ~= Ellipsoide_2_ancien) &  Ellipsoide ~= Ellipsoide_ancien
       essai = -1;
       output = 1;
     end
    
end

%%

% Mise � jour de l'�v�nement
nouveau_evenement= evenement;

%%
% Cas o� une collision a �t� rat�e
if isempty(Ellipsoide_1_ancien) == 0
    
    % Calcul du nouveau point de contact
    
    % Calcul de A et B au temps de collision
    A = matrice(dt^2,Rayon_A,Position_A,Vitesse_A);
    B = matrice(dt^2,Rayon_B,Position_B,Vitesse_B);
    % Calcul des vecteur propre de A^-1*B
    [P,D] = eig(B,A);
    D = real(D);
    % Recherche du vecteur propre qui correspond � la plus petite racine
    % (i.e les racine n�gatifs
    [min3, indice3] = min(min(D));
    vecteur = real(P(:,indice3));
    
    % Remplissage de l'ensemble de cellule Temps_prochain
    Temps_prochain = cell(1,5);
    
    Temps_prochain{1} = dt^2;
    Temps_prochain{2} = Ellipsoide_2_ancien;
    Temps_prochain{3} = Ellipsoide_1_ancien;
    Temps_prochain{4} = 'collision';
    Temps_prochain{5} = vecteur(1:3)'/vecteur(4);
    
    % Cas o� une sortie de cube a �t� rat�e
elseif isempty(Ellipsoide_ancien) == 0
    
    % Remplissage de l'ensemble de cellule Temps_prochain
    Temps_prochain = cell(1,4);
    
    Temps_prochain{1} = dt^2;
    Temps_prochain{2} = Ellipsoide_ancien;
    % Code binaire
    temp = zeros(1,6);
    temp(2*Face_ancien+Cote_ancien-1) = 1;
    
    Temps_prochain{3} = temp;
    Temps_prochain{4} = 'cube';
    
end
