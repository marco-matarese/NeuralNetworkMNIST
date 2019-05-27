function [neuralNetwork, trainingSetErrors, validationSetErrors] = trainNetworkGradientDescent(neuralNetwork, trainingSetInput, trainingSetTargets, validationSetInput, validationSetTargets, epochs, learningType, E, eta, softmax, printFlag)
% Addestramento della rete neurale con l'algoritmo discesa del gradiente.
% 
% Parametri di input:
%   neuralNetwork : Rete neurale istanziata con la funzione newFFMLNeuralNetwork.
%   trainingSetInput : Matrice di valori tale che la riga i-sima rappresenta un
%                      input per la rete neurale.
%   trainingSetTargets : Matrice di valori tale che la riga i-sima rappresenta il target
%                       da ottenere rispetto ai valori di output generati dalla rete neurale.
%   validationSetInput : Matrice di valori tale che la riga i-sima rappresenta un
%                        input per la rete neurale.
%   validationSetTargets : Matrice di valori tale che la riga i-sima rappresenta il target
%                          da ottenere rispetto ai valori di output generati dalla rete neurale.
%   epochs : Numero di epoche su cui addestrare la rete.
%   learningType : Specificare con una stringa in {ONLINE, BATCH} il tipo di 
%                  apprendimento da fare.
%   E : Puntatore alla funzione da utilizzare per il calcolo dell'errore da utilizzare.
%   eta: Numero reale piccolo che rappresenta lo scostamento di interesse
%        rispetto la derivata.
%   softmax: Parametro booleano: se uguale a true, all'output della
%            rete (dopo la forward propagation) verra' applicato il softmax; se falso, no. 
%   printFlag: Impostare a true se si desidera stampare a video i
%              valori degli errori calcolati rispetto al training set
%              ed al validation set.
%
% Parametri di output
%   neuralNetwork : Rete neurale addestrata sul training set.
%   trainingSetErrors : Array di errori tale che l'i-simo elemento 
%                       rappresenta l'errore sul training set relativo
%                       all'epoca i.
%   validationSetErrors : Array di errori tale che l'i-simo elemento 
%                         rappresenta l'errore sul validation set relativo
%                         all'epoca i.

    % Controllo se il numero di colonne della matrice trainingSetInput coincide con il
    % numero di nodi che sono presenti nel layer di input.
    if size(trainingSetInput, 2) ~= neuralNetwork.inputSize
        error("The TRAINING input size is not correct.\nThe number of nodes in the input layer is %d, but the TRAINING input size of the network is %d.", size(X, 2), neuralNetwork.inputSize);
    end
    
    % Controllo se il numero di colonne della matrice validationSetInput coincide con il
    % numero di nodi che sono presenti nel layer di input.
    if size(validationSetInput, 2) ~= neuralNetwork.inputSize
        error("The VALIDATION input size is not correct.\nThe number of nodes in the input layer is %d, but the VALIDATION input size of the network is %d.", size(X, 2), neuralNetwork.inputSize);
    end
    
    % Controllo se il numero di colonne della matrice trainingSetTargets coincide con il
    % numero di nodi che sono presenti nel layer di output.
    if size(trainingSetTargets, 2) ~= neuralNetwork.outputSize
        error("The TRAINING output size is not correct.\nThe number of nodes in the output layer is %d, but the TRAINING output size of the network is %d.", size(T, 2), neuralNetwork.outputSize);
    end
    
    % Controllo se il numero di colonne della matrice validationSetTargets coincide con il
    % numero di nodi che sono presenti nel layer di output.
    if size(validationSetTargets, 2) ~= neuralNetwork.outputSize
        error("The VALIDATION output size is not correct.\nThe number of nodes in the output layer is %d, but the VALIDATION output size of the network is %d.", size(T, 2), neuralNetwork.outputSize);
    end
    
    % Controllo se il numero di vettori target coincide con il numero di
    % vettori input che sono stati passati alla rete per il training set.
    if size(trainingSetInput, 1) ~= size(trainingSetTargets, 1)
        error("The number of TRAINING target vectors must be equal to the number of TRAINING input vectors.\nThe number of TRAINING target vectors is %d, but the number of TRAINING input vectors is %d.", size(trainingSetTargets, 1), size(trainingSetInputs, 1));
    end
    
    % Controllo se il numero di vettori target coincide con il numero di
    % vettori input che sono stati passati alla rete per il validation set.
    if size(validationSetInput, 1) ~= size(validationSetTargets, 1)
        error("The number of VALIDATION target vectors must be equal to the number of VALIDATION input vectors.\nThe number of VALIDATION target vectors is %d, but the number of VALIDATION input vectors is %d.", size(trainingSetTargets, 1), size(trainingSetInputs, 1));
    end
    
    % Inizializzazione degli array per gli errori (uno per ogni epoca).
    trainingSetErrors = zeros(1, epochs);
    validationSetErrors = zeros(1, epochs);
    
    % CRITERIO DI STOP + SALVO BEST NET
    bestValidationSetError = realmax;
    bestNeuralNetwork = neuralNetwork;
    minEpochs = floor(epochs/3);
    
    % Omologazione dei caratteri passati in input per il tipo di
    % addestramento.
    learningType = upper(learningType);
    
    % Addestramento della rete per ogni epoca.
    for epoch = 1 : epochs
        % Stampo a video l'epoca di addestramento attuale.
        fprintf('EPOCH %d.\n', epoch);
        
        % L'errore sul training set e sul validation set viene valutato
        % contestualmente all'aggiornamento della rete. Devo quindi salvare
        % la rete prima dell'aggiornamento.
        prevNetwork = neuralNetwork;
        
        % Controllo il tipo di apprendimento da fare.
        if strcmp(learningType, 'ONLINE')
            [neuralNetwork, trainingSetErrors(epoch), validationSetErrors(epoch)] = onlineGradientLearning(neuralNetwork, trainingSetInput, trainingSetTargets, validationSetInput, validationSetTargets, E, eta, softmax, printFlag);
        elseif strcmp(learningType, 'BATCH')
            [neuralNetwork, trainingSetErrors(epoch), validationSetErrors(epoch)] = batchGradientLearning(neuralNetwork, trainingSetInput, trainingSetTargets, validationSetInput, validationSetTargets, E, eta, softmax, printFlag);
        else
            error('The algorithm does not support this kind of learning type (use BATCH or ONLINE).');
        end
        
        % Se l'errore sul validation aumenta per un numero di epoche fissato, 
        % si evita l'overfitting terminando, a patto che siano stati
        % eseguiti un numero minimo di epoche.
        if validationSetErrors(epoch) < bestValidationSetError 
            incrementErrorCount = 0;
            bestValidationSetError = validationSetErrors(epoch);
            bestNeuralNetwork = prevNetwork;
        else
            if epoch>=minEpochs
                incrementErrorCount = incrementErrorCount + 1;
                if incrementErrorCount == 20
                    break;
                end
            end
        end 
    end
    
    % Se il learning termina prematuramente gli array degli errori va
    % ridotto.
    if epoch < epochs
        trainingSetErrors = trainingSetErrors(1:epoch);
        validationSetErrors = validationSetErrors(1:epoch);
    end
    
    % recupero la rete che ha registrato l'errore sul validation set minimo
    neuralNetwork = bestNeuralNetwork;
end

