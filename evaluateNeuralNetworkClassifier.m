function [totalAccuracy] = evaluateNeuralNetworkClassifier(output, labels)
% Valuta le prestazioni della rete neurale.
% 
% Parametri di input
%   output : matrice NxC di output della rete neurale dopo forward
%             propagation con N input, dove C e' il numero di classi.
%             Per ogni riga la risposta delle rete viene considerata la
%             classe a cui corrisponde il valore maggiore sulla riga.
%   labels : matrice NxC, dove labels(i,j)=1 se l'i-esimo elemento
%            appartiene alla j-esima classe, 0 altrimenti.
%
% Parametri di output
%   totalAccuracy : accuracy della rete, definita come il numero di dati in
%                   input associati alla classe corretta sul numero di dati
%                   totali.

    % Controllo dei parametri di ingresso.
    if (size(output,1) ~= size(labels,1)) || (size(output,2) ~= size(labels,2))
        error("The sizes of input parameters must be equal"); 
    end
    
    % Ottengo le risposte di classificazione della rete.
    classificationAnswer = extractClassificationAnswer(output);
    
    % Calcolo le risposte corrette.
    correct = nnz(classificationAnswer .* labels);
    
    % Calcolo dell'accuracy.
    totalAccuracy = correct/size(output,1);
end
    