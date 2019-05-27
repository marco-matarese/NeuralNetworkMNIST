function neuralNetwork = updateWeights(neuralNetwork, derivativeB, derivativeW, eta)
% Aggiorna i pesi, ed il bias, della rete con i valori delle
% derivate calcolate ed un parametro eta.
% 
% Parametri di input
%   neuralNetwork : Rete neurale ritornata dalla funzione backPropagation.
%   derivativeB : Primo parametro ritornato dalla funzione computeWeightsDerivative.
%   derivativeW : Secondo parametro ritornato dalla funzione computeWeightsDerivative.
%   eta : Numero reale piccolo che rappresenta lo scostamento di interesse
%         rispetto la derivata.
% Parametri di output
%   neuralNetwork : Struct ritornata da backPropagation con pesi e bias aggiornati.

    % Per ogni hidden layer e layer di output della rete, aggiorna pesi e
    % bias con uno scostamento moltiplicativo eta dalla derivata.
    for l = 1 : neuralNetwork.numOfHiddenLayers+1
        neuralNetwork.b{l} = neuralNetwork.b{l} - (eta * derivativeB{l});
        neuralNetwork.W{l} = neuralNetwork.W{l} - (eta * derivativeW{l});
    end
end

