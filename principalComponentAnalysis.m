function [coefficients, scores] = principalComponentAnalysis(input)
% principalComponentAnalysis e' una funzione che applica la PCA alla matrice
% passata in input.
%
% Parametri di input
%   input : e' una matrice che ha per righe i vari esempi e per colonne le
%           componenti di ogni esempio.
%
% Parametri di output
%   coefficients : e' una matrice le cui colonne rappresentano un autovettore
%                  della matrice di covarianza. Le colonne sono ordinate 
%                  per valore decrescente rispetto gli autovalori corrispondenti.
%   scores : e' l'array di autovalori della matrice di covarianza, ordinato
%            per valori decrescenti. Gli autovalori sono normalizzati, cioe'
%            ogni autovalore e' stato diviso per la somma degli autovalori,
%            cosi' la loro somma fa uno. 

    % Calcola la media su ogni riga
    inputMean = mean(input, 1);
    
    % Costruzione della matrice di covarianza
    covarianceMatrix = (input - inputMean)' * (input - inputMean);
    
    % Calcola autovettori ed autovalori della matrice di covarianza
    [eigenVectors, eigenValues] = eig(covarianceMatrix);
    
    % Ordinamento degli autovettori rispetto ai corrispondenti autovalori,
    % per valori descrescenti
    [eigenValues, permutation] = sort(diag(eigenValues), 'descend');
    
    % Normalizzazione degli autovalori
    scores = eigenValues / sum(eigenValues);
    
    % Autovettori ordinati
    coefficients  = eigenVectors(:, permutation);
end

