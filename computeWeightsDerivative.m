function [derivativeB, derivativeW] = computeWeightsDerivative(neuralNetwork)
% Calcola le derivate parziali della funzione di errore
% utilizzata nella back propagation rispetto ai pesi, ed al bias, di ogni
% layer della rete.
%
% Parametri di input
%   neuralNetwork : Rete neurale ritornata dalla funzione backPropagation.
%
% Parametri di output
%   derivativeB : Array cell di array tale che l'i-simo array
%                 contiene le derivate parziali della funzione di errore
%                 rispetto ai bias del layer i. Ogni colonna j di ogni array corrisponde
%                 alla suddetta derivata parziale, ma del nodo j del layer i.
%   derivativeW : Array cell di matrici tale che l'i-sima matrice
%                 contiene le derivate parziali della funzione di errore
%                 rispetto ai pesi sulle connessioni tra il layer i ed il layer i-1. 

    % Calcolo delle derivate parziali per il primo hidden layer. Poiche'
    % delta{1} avra' dimensione |numRigheX|x|numNodiLayerInput|, e'
    % necessario ottenere una derivata per ogni nodo, e quindi aggregare i
    % valori di delta{1} per riga.
    derivativeB{1} = sum(neuralNetwork.delta{1}, 1);
    % Si ricorda che si hanno piu' vettori di input da considerare alla
    % volta (difatti ci saranno, per ogni layer, tanti vettori di delta
    % quanti sono i vettori passati in input nella forwardPropagation). La
    % dimensione di delta{1} e' |numRigheX|x|numNodiHidden1|, mentre la
    % dimensione di X e' |numRigheX|x|numNodiInputLayer|. Allora per procedere 
    % con la moltiplicazione e' necessario trasporre una delle due matrici: 
    % trasponendo delta{1} si ottiene una matrice di dimensione 
    % |numNodiHidden1|x|numRigheX|, e quindi si puo' procedere al prodotto. 
    % La matrice risultante avra' dimensione |numNodiHidden1|x|numNodiInputLayer|
    % che coincide con la dimensione della matrice che rappresenta le
    % connessioni tra il layer di input ed il primo hidden layer.
    derivativeW{1} = neuralNetwork.delta{1}' * neuralNetwork.X;
    
    % Calcolo le derivate parziali per i restanti hidden layer e layer di
    % output.
    for l = 2 : neuralNetwork.numOfHiddenLayers+1
        derivativeB{l} = sum(neuralNetwork.delta{l}, 1);
        % Stesso ragionamento del calcolo sul primo layer hidden: si
        % calcolano le derivate parziali dei pesi posizionati sulle
        % connessioni tra il layer l (attuale) ed il layer precedente (l-1).
        derivativeW{l} = neuralNetwork.delta{l}' * neuralNetwork.z{l-1};
    end            
end
