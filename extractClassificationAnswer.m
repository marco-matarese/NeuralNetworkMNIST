function classificationAnswer = extractClassificationAnswer(output)
% Estrae le risposte di classificazione di una rete neurale.
%
% Parametri di input
%   outputs : matrice NxC di output della rete neurale dopo forward
%             propagation con N input, dove C e' il numero di classi.
%             Per ogni riga la risposta delle rete viene considerata la
%             classe a cui corrisponde il valore maggiore sulla riga.
%
% Parametri di output
%   classificationAnswer : matrice NxC tale che l'elemento (i,j) e' 1 se
%                          l'i-esimo dato in input appartiene alla classe
%                          j, 0 altrimenti.

    % Calcolo le risposte effettive della rete.
    classificationAnswer = zeros(size(output,1), size(output,2));
    for i = 1 : size(output,1)
        % Trovo la classe a cui corrisponde il valore di uscita maggiore
        % per l'i-esimo elemento.
        [~, argmax] = max(output(i,:));
        % La identifico come risposta della rete.
        classificationAnswer(i,argmax) = 1;
    end
    
end
