function plotErrors(trainingSetErrors, validationSetErrors)
% Creazione del grafico contenente gli andamenti dei grafici per le
% funzioni di errore calcolate sul training set e validation set.

% Parametri di input
%   trainingSetErrors : errori ottenuti ad ogni epoca sul training set.
%   validationSetErrors : errori ottenuti ad ogni epoca sul validation set.

    % Calcolo e disposizione dei grafici
    plot(trainingSetErrors);
    hold
    plot(validationSetErrors);
    legend('Error on the training set', 'Error on the validation set');
end

