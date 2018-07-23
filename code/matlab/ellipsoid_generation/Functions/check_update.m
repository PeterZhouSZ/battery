function [check] = check_update(check,evenement,ancien_evenement,Temps_prochain)
%
%% Description :
%  Cette fonction permet d'identifier les ellipsodes à checker pour la
%  prochaine itération
%
%% Variable(s) d'entrée :
%  check            : ensemble de cellule contenant les matrices temps_ellipsoides et ellipsoides_check
%  evenement        : ensemble de cellule contenant l'ensemble des ellipsoides actuel
%  ancien_evenement : ensemble de cellule contenant l'ensemble des ellipsoides à l'itération précédente
%  Temps_prochain   : Ensemble de cellule contenant le temps du prochain
%                     de evenement et le ou les ellipsoides impliquées
%% Variable de sortie:
%  check : ensemble de cellule contenant les matrices temps_ellipsoides et ellipsoides_check
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1};
Partenaire = evenement{7};
Identite_ancien = ancien_evenement{1};
L1 = length(Identite);
nb_Ellipsoides = size(check{2,1},1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% COLLISION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Mise à jour de la matrice temps_ellipsoides_1
temp = check{1,1} - Temps_prochain{1};
temps_ellipsoides_1 = zeros(L1,L1);
ellipsoides_check_1 = zeros(L1,L1);
% Les nb_Ellipsoides x nb_Ellipsoides termes n'ont pas changé, donc
% transfert automatique
temps_ellipsoides_1(1:nb_Ellipsoides,1:nb_Ellipsoides) = temp(1:nb_Ellipsoides,1:nb_Ellipsoides);

% Transfert des autres termes
for k = nb_Ellipsoides+1:L1
    
    % Recherche dans l'ancien ensemble de cellule
    u = (Identite_ancien == Identite(k));
    
     
    if sum(u) ~= 0
        % Si trouvés
        for l = 1:k-1;
            v = (Identite_ancien == Identite(l));
            if sum(v) ~=0
                temps_ellipsoides_1(l,k) = temp(v,u);
            end
        end
        
    else
        % Si non trouvés, les nouveaux ellipsoides doivent être vérifiés à
        % la prochaine itération.
        ellipsoides_check_1(1:k-1,k) = 1;
        ellipsoides_check_1(k,k+1:end) = 1;
    end
    
end


% Mise à jour de la matrice elipsoides_check_1
if isequal(Temps_prochain{4},'collision') == 1
    
    % Identification des ellipsoides qui sont entrés en collision
    m = Temps_prochain{2};
    n = Temps_prochain{3};
    
    % Recherche de ces ellipsoides dans l'ensemble de cellule
    i = find(Identite == m);
    j = find(Identite == n);
    
    % Tous les pairs contenant ces 2 ellipsoides doivent êtres vérifiés à
    % la prochaine itération
    ellipsoides_check_1(1:i-1,i) = 1;
    ellipsoides_check_1(i,i+1:end) = 1;
    ellipsoides_check_1(1:j-1,j) = 1;
    ellipsoides_check_1(j,j+1:end) = 1;
    
    % Recherche des partenaires de ces ellipsoides
    partenaire_1 = Partenaire(i,:);
    partenaire_2 = Partenaire(j,:);
    
    % Tous les partenaires de l'ellipsoide m doivent être vérifiés à la
    % prochaine itération
    for m = 1:length(partenaire_1);
        i = find(Identite == partenaire_1(m));
        ellipsoides_check_1(1:i-1,i) = 1;
        ellipsoides_check_1(i,i+1:end) = 1;
    end
    
    % Tous les partenaires de l'ellipsoide n doivent être vérifiés à la
    % prochaine itération
    for n = 1:length(partenaire_2);
        j = find(Identite == partenaire_2(n));
        ellipsoides_check_1(1:j-1,j) = 1;
        ellipsoides_check_1(j,j+1:end) = 1;
    end
    
end


% Mise à jour de l'ensemble de cellule check en formant des matrice
% symétriques
check{1,1} = temps_ellipsoides_1 + ellipsoides_check_1';
check{1,2} = ellipsoides_check_1 + ellipsoides_check_1';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CUBE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mise à jour de la matrice temps_ellipsoides_2
check{2,1} = check{2,1} - Temps_prochain{1};

% Mise à jour de la matrice ellipsoides_check_2
check{2,2} = zeros(nb_Ellipsoides,6);

if isequal(Temps_prochain{4},'collision') == 1
    
    % Identification des ellipsoides qui sont entrés en collision
    m = Temps_prochain{2};
    n = Temps_prochain{3};
    
    % Recherche de ces ellipsoides dans l'ensemble de cellule
    i = find(Identite == m);
    j = find(Identite == n);
    
    % Recherche des partenaires de ces ellipsoides
    partenaire_1 = Partenaire(i,:);
    partenaire_2 = Partenaire(j,:);
    
    % Identification des ellipsoides originaux (et non les images
    % périodiques)
    i = min([i,partenaire_1(partenaire_1>0)]);
    j = min([j,partenaire_2(partenaire_2>0)]);
    
    % Les ellipsoides doivent être checké à la prochaine itérations
    check{2,2}(i,:) = 1;
    check{2,2}(j,:) = 1;
    
elseif isequal(Temps_prochain{4},'cube') == 1
    
    % Identification de l'ellipsodide qui est sorti du cube
    p = Temps_prochain{2};
    
    % Identification du code binaire
    code = Temps_prochain{3};
    q = (code == 1);
    
    % Cet ellipsode doit être checké à la prochaine itérations
    check{2,2}(p,q) = 1;
    
end


