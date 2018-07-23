function code_2 = opposer(code_1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function possibilite = permuter(code)
%
%% Description :
% Cette fonction permet de donner l'opposé d'un code binaire
% ex: oppose([1 0 0 0 1 0]) = [0 1 0 0 0 1] , oppose([0 1 1 0 0 1]) = [1 0 0 1 1 0]
%
%% Variable(s) d'entrée :
%  code_1 : Code binaire d'entrée
% 
%% Variable de sortie:
%  code_2 : code binaire de sortie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Initialisation de code_2
code_2 = zeros(1,6);

for i = 1:2:length(code_1)-1
    
    if code_1(i) == 1 || code_1(i+1) ==1
        
        % Permutation des chiffres
        code_2(i) = code_1(i+1);
        code_2(i+1) = code_1(i);
        
    else
        
        % Code demeure inchangé
        code_2(i:i+1) = code_1(i:i+1);
        
    end
    
end

