function [coefficients, lastPrincipalComponentIndex] = applyPrincipalComponentAnalysis(input, threshold)
% Applica la PCA alla matrice di input. Successivamente, vengono calcolati
% quali sono i nuovi valori da mantenere nella nuova matrice trasformata,
% con l'approssimazione scelta dall'utente.
%
% Parametri di input
%   input : Matrice da sottoporre alla principal component analysis
%   threshold : Percentuale di approssimazione
%
% Parametri di output
%   coefficients : e' una matrice le cui colonne rappresentano un autovettore
%                  della matrice di covarianza. Le colonne sono ordinate 
%                  per valore decrescente rispetto gli autovalori corrispondenti.
%                  Vengono prese solo le colonne 1:lastPrincipalComponentIndex.
%   lastPrincipalComponentIndex : Indice che rappresenta l'ultima colonna
%                                 che abbiamo mantenuto nella nuova matrice 
%                                 dei coefficienti.

    % Calcolo della PCA
    [coefficients, scores] = principalComponentAnalysis(input);
    
    % Utilizza la varianza per selezionare il threshold% delle colonne da
    % mantenere nella nuova matrice dei coefficienti
    currentCoverage = 0.0;
    lastPrincipalComponentIndex = 1;
    while (lastPrincipalComponentIndex <= size(scores,1)) && (currentCoverage <= threshold)
        currentCoverage = currentCoverage + scores(lastPrincipalComponentIndex);
        lastPrincipalComponentIndex = lastPrincipalComponentIndex + 1;
    end
    
    % Nuova matrice dei coefficienti
    coefficients = coefficients(:, 1:lastPrincipalComponentIndex);
end

