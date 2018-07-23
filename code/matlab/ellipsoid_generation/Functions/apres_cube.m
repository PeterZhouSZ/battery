function [nouveau_evenement] = apres_cube(evenement, ellipsoide, code, nb_Ellipsoides)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [nouveau_evenement] = apres_cube(evenement, ellipsoide, code, nb_Ellipsoides)
%
%% Description :
%  Cette fonction permet de mettre � jour un ellipsoide qui a traverse une ou
%  plusieurs face d'un cube. Elle permet �galement de cr�er les ellipsoides
%  partenaires pour satisfaire les conditions de p�riodicit�
%
%% Variable(s) d'entr�e :
%  evenement:      ensemble de cellule contenant l'ensemble des ellipsoides
%  sph�re:         Num�ro d'identification de l'ellipsoide qui a travers� une ou
%                  plusieurs face d'un cube
%  code:           Code binaire permettant d'identifier de quel face du cube
%                  est sortie l'ellipsoide
%  nb_Ellipsoides: Nombre d'ellipsoides

%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des
%  ellipsoide mis � jour.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1};
Temps = evenement{2};
Rayon = evenement{3};
Position = evenement{4};
Vitesse = evenement{5};
Code = evenement{6};
Partenaire = evenement{7};

% Recherche de l'ellipsoide dans l'ensemble de cellule nouveau_evenement
m = find(Identite == ellipsoide);

% Identifiaction des partenaires
partenaire = Partenaire(m,:);


% Copie de l'ensemble de cellule
nouveau_evenement = evenement;

% Effacer tous les partenaires ant�rieur de l'ellipsoide
L = size(Identite,1);
if sum(partenaire) ~= 0
    for k=(nb_Ellipsoides+1):L
        if Identite(k) == partenaire(1)
            
            Identite = Identite([1:k-1,k+sum(partenaire~=0):end]);
            Temps = Temps([1:k-1,k+sum(partenaire~=0):end],:);
            Rayon = Rayon([1:k-1,k+sum(partenaire~=0):end],:);
            Position = Position([1:k-1,k+sum(partenaire~=0):end],:);
            Vitesse = Vitesse([1:k-1,k+sum(partenaire~=0):end],:);
            Code = Code([1:k-1,k+sum(partenaire~=0):end],:);
            Partenaire = Partenaire([1:k-1,k+sum(partenaire~=0):end],:);
            break
            
        end
    end
end


% Mise � jour du code binaire de l'ellipsoide
for p = 1:length(code)
    Code(m,p) = Code(m,p) + code(p);
end

% Cr�ation des partenaires
[Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire] = creer_partenaire(Identite, Temps, Rayon, Position, Vitesse, Code, Partenaire, m);

% Insertion des matrices dans l'ensemble de cellule
nouveau_evenement{1} = Identite;
nouveau_evenement{2} = Temps;
nouveau_evenement{3} = Rayon;
nouveau_evenement{4} = Position;
nouveau_evenement{5} = Vitesse;
nouveau_evenement{6} = Code;
nouveau_evenement{7} = Partenaire;
