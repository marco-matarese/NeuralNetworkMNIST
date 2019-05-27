function [neuralNetwork, trainingSetE, validationSetE] = batchGradientLearning(neuralNetwork, trainingSetInput, trainingSetTargets, validationSetInput, validationSetTargets, E, eta, softmax, printFlag)
% Singolo passo di training della rete neurale con approccio di learning di 
% tipo batch.
%
% Parametri di input
%   neuralNetwork : Rete neurale istanziata con la funzione newFFMLNeuralNetwork.
%   trainingSetInput : Matrice di valori tale che la riga i-sima rappresenta un
%                      input per la rete neurale.
%   trainingSetTargets : Matrice di valori tale che la riga i-sima rappresenta il target
%                        da ottenere rispetto ai valori di output generati dalla rete neurale.
%   validationSetInput : Matrice di valori tale che la riga i-sima rappresenta un
%                        input per la rete neurale.
%   validationSetTargets : Matrice di valori tale che la riga i-sima rappresenta il target
%                          da ottenere rispetto ai valori di output generati dalla rete neurale.
%   E : Puntatore alla funzione da utilizzare per il calcolo dell'errore da utilizzare.
%   eta : Numero reale piccolo che rappresenta lo scostamento di interesse
%         rispetto la derivata.
%   softmax : Parametro booleano: se uguale a true, all'output della
%             rete (dopo la forward propagation) verra' applicato il softmax; 
%             se falso, no.
%   printFlag : Impostare a true se si desidera stampare a video i
%               valori degli errori calcolati rispetto al training set
%               ed al validation set.
%
% Parametri di output
%   neuralNetwork : Rete neurale con input il training set, ritornata dalla funzione
%                   backPropagation, con pesi e bias aggiornati con le derivate.
%   trainingSetE : Errore relativo al training set.
%   validationSetE : Errore relativo al validation set.

    % Inizializzazione degli errori a zero.
    trainingSetE = 0;
    validationSetE = 0;
    
    % Forward propagation per il training set.
    neuralNetworkTraining = forwardPropagation(neuralNetwork, trainingSetInput, softmax);
    % Forward propagation per il validation set.
    neuralNetworkValidation = forwardPropagation(neuralNetwork, validationSetInput, softmax);
        
    % Calcolo dell'errore per il training set.
    trainingSetE = sum(E(neuralNetworkTraining.z{neuralNetworkTraining.numOfHiddenLayers+1}, trainingSetTargets));
    % Calcolo dell'errore per il validation set.
    validationSetE = sum(E(neuralNetworkValidation.z{neuralNetworkValidation.numOfHiddenLayers+1}, validationSetTargets));
        
    % Controllo se l'utente ha deciso di stampare a video gli errori.
    if printFlag
        fprintf('Error on the TRAINING set: %f.\nError on the VALIDATION set: %f.\n', trainingSetE, validationSetE);
    end
        
    % Calcolo della back propagation per la rete sul training set.
    neuralNetworkTraining = backPropagation(neuralNetworkTraining, trainingSetTargets, E);        
        
    % Calcolo delle derivate per la rete sul training set.
    [trainingDerivB, trainingDerivW] = computeWeightsDerivative(neuralNetworkTraining);
        
    % Aggiornamento dei pesi per la rete sul training set.
    neuralNetworkTraining = gradientDescent(neuralNetworkTraining, trainingDerivB, trainingDerivW, eta);
    
    % Ritorno
    neuralNetwork = neuralNetworkTraining;
end



