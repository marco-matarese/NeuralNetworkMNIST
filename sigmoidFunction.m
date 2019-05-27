function y = sigmoidFunction(a, derivateFlag)
% Calcola la funzione sigmoide sull'input a.
% 
% Parametri di input:
%   a : Un valore numerico oppure un array di valori numerici.
%
% Parametri di input opzionali:
%   derivateFlag : Se questo parametro viene passato alla funzione,
%                  verra' calcolata la derivata della funzione sigmoide.
%
% Parametri di output:
%   y : Numero che rappresenta la valutazione della funzione sigmoide (o
%       della sua derivata) rispetto l'input a.

    % Se il parametro derivateFlag viene passato alla funzione.
    if exist('derivateFlag','var')
        y = sigmoidFunction(a) .* (1-sigmoidFunction(a)); 
    % Se il parametro derivateFlag non viene passato alla funzione.
    else
        y = 1.0 ./ (1.0 + exp(-a)); 
    end
end

