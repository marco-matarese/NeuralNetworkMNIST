function y = crossEntropyFunction(X, Y, derivateFlag)
% Calcola la funzione cross entropy sull'input X e Y.
% 
% Parametri di input:
%   X : Un valore numerico oppure un array di valori numerici.
%   Y : Un valore numerico oppure un array di valori numerici.
%
% Parametri di input opzionali:
%   derivateFlag : Se questo parametro viene passato alla funzione,
%                  verra' calcolata la derivata della funzione cross entropy.
%
% Parametri di output:
%   y : Numero che rappresenta la valutazione della funzione cross entropy 
%       (o della sua derivata) rispetto l'input X e Y.

    % Se il parametro derivateFlag viene passato alla funzione.
    % La funzione è stata implementata considerando che venisse usata assieme
    % all'applicazione del softmax sugli output della rete. Senza utilizzare il
    % softmax sarebbe dovuto essere -Y/X.
    if exist('derivateFlag','var')
        y = X - Y;
    % Se il parametro derivateFlag non viene passato alla funzione.
    else
        % Per i valori maggiori di zero, viene applicata la funzione standard.
        Y(X > 0) = Y(X > 0) .*  log(X(X > 0));
        % Per i valori che sono zero è un problema calcolarne poi il logaritmo. 
        % Allora andiamo ad assegnare un valore negativo molto grande.
        Y(X == 0) = Y(X == 0) * (-708);   % log del minimo reale rappresentabile in MATLAB
        
        y = - sum(Y);
    end
end