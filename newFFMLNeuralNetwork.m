function neuralNetwork = newFFMLNeuralNetwork(inputSize, outputSize, outputActivation, hiddenLayers, infWeights, supWeights)
% Crea una rete neurale feed-forward multi-strato.
% 
% Parametri di input
%   inputSize : Numero di nodi del layer di input.
%   outputSize : Numero di nodi del layer di output.
%   outputActivation : Puntatore alla funzione da utilizzare per il layer di output.
%   hiddenLayers : E' un array in cui ogni elemento e' una struct avente i seguenti campi:
%                  [4.1] layerSize: Numero di nodi dell'hidden layer.
%                  [4.2] activationFunction: Puntatore alla funzione da
%                        utilizzare per questo hidden layer.
%   infWeights : Estremo inferiore dell'intervallo dei valori con cui
%                riempire casualmente i pesi della rete.
%   supWeights:  Estremo superiore dell'intervallo dei valori con cui
%                riempire casualmente i pesi della rete.
%
% Parametri di output
%   inputSize : Numero di nodi nel layer di input.
%   outputSize : Numero di nodi nel layer di output.
%   numOfHiddenLayers : Numero di hidden layers.
%   numOfLayers : Numero di layer totali.
%   m : Array di valori interi tale che l'elemento i-simo rappresenta il
%       numero di nodi presenti nell'hidden layer i.
%   b : Cell array di matrici monodimensionali tale che l'elemento
%       i-simo rappresenta i valori dei bias del layer i.
%   W : Cell array di matrici bidimensionali tale che l'elemento i-simo
%       rappresenta i pesi che si trovano sulle connessioni tra il layer
%       i ed il layer i-1 della rete.
%   g : Array contenente puntatori a funzione tale che l'elemento i-simo
%       rappresenta la funzione di attivazione del layer i.

    % Controllo se gli elementi dell'array hiddenLayers sono rappresentati
    % per riga o per colonna. Se gli elementi sono rappresentati per riga,
    % traspongo l'array per utilizzare la notazione che prevede gli elementi 
    % rappresentati per colonne (per omologazione futura).
    if size(hiddenLayers, 1) > size(hiddenLayers, 2)
        hiddenLayers = hiddenLayers';
    end
    
    % Inizializzazioni dirette.
    neuralNetwork.inputSize = inputSize;
    neuralNetwork.outputSize = outputSize;
    neuralNetwork.numOfHiddenLayers = size(hiddenLayers, 2);
    neuralNetwork.outputActivation = outputActivation;
    
    % Il numero di layer totali e' dato dal numero di hidden layer piu' il
    % layer di input ed il layer di output.
    neuralNetwork.numOfLayers = size(hiddenLayers, 2) + 2;
    
    % Aggiorno l'array m con il numero di nodi presenti all'interno di ogni
    % hidden layer.
    for l = 1 : neuralNetwork.numOfHiddenLayers
        neuralNetwork.m(l) = hiddenLayers(l).layerSize;
    end
    
    % Generazione dei pesi e dei bias con valori casuali nell'intervallo 
    % [infWeights, supWeights] tra il layer di input ed il primo hidden layer
    % della rete, ed inizializzazione della funzione di attivazione 
    % del primo hiddem layer con quella passata in input.
    neuralNetwork.b{1} = (supWeights-infWeights) .* rand(1, neuralNetwork.m(1)) + infWeights;
    neuralNetwork.W{1} = (supWeights-infWeights) .* rand(neuralNetwork.m(1), inputSize) + infWeights;
    neuralNetwork.g{1} = hiddenLayers(1).activationFunction;
    
    % Se la rete contiene un numero di hidden layer che e' maggiore di uno,
    % si generano valori casuali per pesi e bias, nell'intervallo [infWeights, supWeights],
    % per le connessioni tra ogni hidden layer. Si inizializza, inoltre, la
    % funzione di attivazione del layer corrente con quella passata in
    % input.
    if neuralNetwork.numOfHiddenLayers > 1
        for l = 2 : neuralNetwork.numOfHiddenLayers
            neuralNetwork.b{l} = (supWeights-infWeights) .* rand(1, neuralNetwork.m(l)) + infWeights;
            neuralNetwork.W{l} = (supWeights-infWeights) .* rand(neuralNetwork.m(l), neuralNetwork.m(l-1)) + infWeights;
            neuralNetwork.g{l} = hiddenLayers(l).activationFunction;
        end
    end
    
    % Generazione dei pesi e dei bias con valori casuali nell'intervallo 
    % [infWeights, supWeights] tra il layer di output e l'ultimo hidden
    % layer della rete, ed inizializzazione della funzione di attivazione
    % del layer di output passata in input.
    neuralNetwork.b{neuralNetwork.numOfLayers-1} = (supWeights-infWeights) .* rand(1, outputSize) + infWeights;
    neuralNetwork.W{neuralNetwork.numOfLayers-1} = (supWeights-infWeights) .* rand(outputSize, neuralNetwork.m(neuralNetwork.numOfHiddenLayers)) + infWeights;
    neuralNetwork.g{neuralNetwork.numOfLayers-1} = outputActivation;
end

