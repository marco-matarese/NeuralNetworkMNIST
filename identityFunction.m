function y = identityFunction(a, derivateFlag)
% Calcola la funzione identita' sull'input a.
% 
% Parametri di input:
%   a : Un valore numerico oppure un array di valori numerici.
%
% Parametri di output
%   derivateFlag : Se questo parametro viene passato alla funzione,
%                  verra' calcolata la derivata della funzione identita'.
%
% Parametri di output
%   y : Numero che rappresenta la valutazione della funzione identita' (o
%       della sua derivata) rispetto l'input a.

    % Se il parametro derivateFlag viene passato alla funzione.
    if exist('derivateFlag','var')
        y = ones(size(a));
    % Se il parametro derivateFlag non viene passato alla funzione.
    else
        y = a; 
    end
end

