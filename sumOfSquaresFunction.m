function y = sumOfSquaresFunction(X, Y, derivateFlag)
% Calcola la funzione somma dei quadrati sull'input X e Y.
% 
% Parametri di input:
%   X : Un valore numerico oppure un array di valori numerici.
%   Y : Un valore numerico oppure un array di valori numerici.
%
% Parametri di input opzionali:
%   derivateFlag : Se questo parametro viene passato alla funzione,
%                  verra' calcolata la derivata della funzione somma dei quadrati.
%
% Parametri di output:
%   y = Numero che rappresenta la valutazione della funzione somma dei quadrati 
%       (o della sua derivata) rispetto l'input X e Y.

    % Se il parametro derivateFlag viene passato alla funzione.
    if exist('derivateFlag','var')
        y = X - Y;
    % Se il parametro derivateFlag non viene passato alla funzione.
    else
        y = 0.5 * sum(((X - Y) .^ 2));
    end
end

