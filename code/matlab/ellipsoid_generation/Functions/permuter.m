function possibilite = permuter(code)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function possibilite = permuter(code)
%
%% Description :
% Cette fonction permet de donner le code binaire de tous les partenaires
% possibles de l'ellipsoide.
%
%% Variable(s) d'entrée :
%  code     : Code binaire permettant d'identifier de quel face du cube
%             est sortie l'ellipsoide
%% Variable de sortie:
%  possibilité : ensemble de cellule contenant l'ensemble des codes permutés
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

m=0;
nb=zeros(1,length(code)/2);

% Calcul du nombre de permutation possible
for i =1:2:length(code)-1
    if code(i) == 1 || code(i+1) == 1
        m=m+1;
        nb(m) = (i+1)/2;
    end
end

nb = nb(1:m);
possibilite = cell( 1,(2^length(nb)) );

% Cas où 1 permutation est possible
if length(nb) == 1
    
    n=0;
    for i=(2*nb(1)-1):(2*nb(1))
        n=n+1;
        possibilite{1,n} = zeros(1,6);
        possibilite{1,n}(i) = 1;
    end
 
% Cas où 3 permutations sont possibles    
elseif length(nb) == 2
    
    n=0;    
    for i=(2*nb(1)-1):(2*nb(1))
        for j=(2*nb(2)-1):(2*nb(2))
            n=n+1;
            possibilite{1,n} = zeros(1,6);
            possibilite{1,n}(i) = 1;
            possibilite{1,n}(j) = 1;
        end
    end

% Cas où 7 permutations sont possibles     
elseif length(nb) == 3
    
    n=0;
    possibilite = cell( 1,(2^length(nb)) );
    
    for i=(2*nb(1)-1):(2*nb(1))
        for j=(2*nb(2)-1):(2*nb(2))
            for k=(2*nb(3)-1):(2*nb(3))
                n=n+1;
                possibilite{1,n} = zeros(1,6);
                possibilite{1,n}(i) = 1;
                possibilite{1,n}(j) = 1;
                possibilite{1,n}(k) = 1;
            end
        end
    end
    
else
    
    possibilite={};
    
end

% Supression de la matrice code dans la liste
for i =1:size(possibilite,2)
    if isequal(possibilite{i},code) ==1
        
        possibilite = possibilite(1,[1:i-1,i+1:end]);
        break
    end
end
