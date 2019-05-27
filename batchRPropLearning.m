function [neuralNetwork, trainingSetE, validationSetE, deltaB, deltaW, trainingDerivB, trainingDerivW] = batchRPropLearning(neuralNetwork, trainingSetInput, trainingSetTargets, validationSetInput, validationSetTargets, E, etaMinus, etaPlus, epoch, derivativeWPrec, derivativeBPrec, deltaWPrec, deltaBPrec, softmax, printFlag)
% Funzione di addestramento con resilient back propagation per una singola
% epoca.
% 
% Parametri di input
%   neuralNetwork : rete neurale istanziata con la funzione newFFMLNeuralNetwork.
%   trainingSetInput : Il training set per addestrare la rete. Matrice di valori 
%                      tale che la riga i-sima rappresenta un input per la rete neurale.
%   trainingSetTargets : Matrice di valori tale che la riga i-sima rappresenta il target
%                       da ottenere rispetto ai valori di output generati dalla rete neurale,
%                       quando prende come input il training set.
%   validationSetInput : Matrice di valori tale che la riga i-sima rappresenta un
%                        input per la rete neurale. Viene utilzzato per
%                        evitare overfitting sul training set.
%   validationSetTargets : Matrice di valori tale che la riga i-sima rappresenta 
%                         il target da ottenere rispetto ai valori di output 
%                         generati dalla rete neurale, quando prende come input il validation set.
%   epochs : Numero di epoche con cui addestare la rete.
%   learningType : Specificare con una stringa in {ONLINE, BATCH} il tipo di 
%                  apprendimento da fare.
%   E : Puntatore alla funzione da utilizzare per il calcolo dell'errore da utilizzare.
%   etaMinus : Numero reale piccolo che rappresenta il fattore moltiplicativo 
%              per gli scostamenti precedenti della matrice dei pesi, quando la 
%              derivata della funzione di errore e' discorde con quella precedente. 
%              Valore consigliato : 0.5.
%   etaPlus : Numero reale piccolo che rappresenta il fattore moltiplicativo rispetto
%             allo scostamento precedente della matrice dei pesi, quando la 
%             derivata della funzione di errore e' concorde con quella precedente. 
%             Valore consigliato : 1.2.
%   derivativeWPrec : Cell array che associa ad ogni layer di pesi della
%                     rete le derivate della funzione di errore rispetto ad
%                     ogni peso, calcolate nell'epoca precedente.
%   derivativeBPrec : Cell array che associa ad ogni layer di pesi della
%                     rete le derivate della funzione di errore rispetto ad
%                     ogni bias, calcolate nell'epoca precedente.
%   deltaWPrec : Cell array che associa ad ogni layer di pesi, la
%                variazione effettuata sui pesi nell'epoca precedente.
%   deltaBPrec : Cell array che associa ad ogni layer di nodi, la
%                variazione effettuata su ogni bias nell'epoca precedente.
%   softmax : Parametro booleano: se uguale a true, all'output della rete 
%             (dopo la forward propagation) verra' applicato il softmax; se falso, no. 
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
%   deltaB : Array cell che associa ad ogni layer di nodi la variazione
%            effettuata per ogni bias.
%   deltaW : Array cell che associa ad ogni layer di pesi la variazione
%            effettuata per ogni peso.
%   trainingDerivB : Array cell che associa ad ogni layer di nodi la
%                    derivata della funzione di errore calcolata rispetto 
%                    ad ogni bias.
%   trainingDerivW : Array cell che associa ad ogni layer di pesi la
%                    derivata della funzione di errore calcolata rispetto
%                    ad ogni peso.

    % Forward propagation per l'intero training set.
    neuralNetworkTraining = forwardPropagation(neuralNetwork, trainingSetInput, softmax);
    % Forward propagation per l'intero validation set.
    neuralNetworkValidation = forwardPropagation(neuralNetwork, validationSetInput, softmax);
    
    % Calcolo dell'errore totale per entrambi i set.
    trainingSetE = sum(E(neuralNetworkTraining.z{neuralNetworkTraining.numOfHiddenLayers+1}, trainingSetTargets));
    validationSetE = sum(E(neuralNetworkValidation.z{neuralNetworkValidation.numOfHiddenLayers+1}, validationSetTargets));
    
    % Eventuale stampa dei valori della funzione di errore.
    if printFlag
        fprintf('Error on the TRAINING set: %f.\nError on the VALIDATION set: %f.\n', trainingSetE, validationSetE);
    end
        
    % Calcolo dei valori dei delta con back propagation.
    neuralNetworkTraining = backPropagation(neuralNetworkTraining, trainingSetTargets, E);        
    
    % Calcolo delle derivate dei pesi e dei bias.
    [trainingDerivB, trainingDerivW] = computeWeightsDerivative(neuralNetworkTraining);
    
    % Aggiornamento della rete con resilient back propagation.
    [neuralNetwork, deltaW, deltaB] = resilientBackPropagation(neuralNetworkTraining, trainingDerivB, trainingDerivW, derivativeBPrec, derivativeWPrec, deltaWPrec, deltaBPrec, etaMinus, etaPlus, epoch);
    
end