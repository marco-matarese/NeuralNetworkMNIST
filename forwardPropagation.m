function neuralNetwork = forwardPropagation(neuralNetwork, X, useSoftmax)
% Propaga la rete neurale dal layer di input verso il layer di output.
% 
% Parametri di input
%   neuralNetwork : Rete neurale istanziata con la funzione newFFMLNeuralNetwork.
%   X : Matrice di valori tale che la riga i-sima rappresenta un
%       input per la rete neurale.
%   useSoftmax : Parametro booleano: se uguale a true, all'output della
%                rete verra' applicato il softmax; se falso, no.
%
% Parametri di output
%   a : Array cell di matrici tale che ogni riga j dell'i-sima matrice rappresenta i
%       valori di input del layer i per il vettore di input j (le
%       colonne sono gli input dei singoli nodi).
%   z : Array cell di matrici tale che ogni riga j dell'i-sima matrice rappresenta i
%       valori di output del layer i per il vettore di input j (le
%       colonne sono gli output dei singoli nodi).
%   X : Matrice tale che la riga i-sima rappresenta l'i-simo vettore di
%       input passato al primo layer della rete.
    
    % Salvo i valori dello strato di input.
    neuralNetwork.X = X;
    
    % Questa variabile rappresenta l'output del layer che viene considerato di volta
    % in volta (si comincia con il layer di input).
    outputOfPreviousLayer = X;
    
    % Per ogni hidden layer, e layer di output, si calcola l'input del layer corrente e
    % l'output del layer corrente.
    for l = 1 : neuralNetwork.numOfHiddenLayers+1
        % Calcolo l'input del layer l. Le connessioni considerate sono
        % quelle tra questo layer ed il layer l-1, quindi la dimensione di
        % W{l} e' |NumNodiLayer_l|x|NumNodiLayer_l-1]. La dimensione di
        % valuesOfCurrentLayer e' diversa, poiche' ogni riga di questa
        % matrice rappresenta l'input calcolato su uno specifico vettore
        % dell' input X. Il numero di colonne di valuesOfCurrentLayer
        % rappresenta il numero di nodi sul layer corrente; in definitiva:
        % |NumVettoriInput|x|NumNodiLayer_l|. Allora per fare il prodotto
        % si puo' trasporre valuesOfCurrentLayer. Cio' che si ottiene da
        % questo prodotto e' una matrice che ha come righe il numero di
        % nodi di input sul layer l e come colonne il numero di vettori
        % su cui e' stato effettuato il prodotto; poiche' la
        % rappresentazione standard e' quella opposta, si effettua una
        % trasposizione sulla matrice risultante dal prodotto.
        % neuralNetwork.a{l} = (neuralNetwork.W{l} * outputOfPreviousLayer')';
        neuralNetwork.a{l} = (outputOfPreviousLayer * neuralNetwork.W{l}');
        % Calcolo dell'output del layer l. La dimensione della matrice
        % dell'input del nodo corrente e' |NumVettoriInput|x|NumNodiLayer_l|
        % (come spiegato all'istruzione precedente). Il bias del layer l ha
        % cardinalita' 1x|NumNodiLayer_l|, cio' che si vuole fare e' sommare
        % il bias di questo layer ad ogni riga della matrice di input. Matlab,
        % automaticamente, trasforma la matrice del bias da
        % 1x|NumNodiLayer_l| a |NumVettoriInput|x|NumNodiLayer_l|, e cioe'
        % applica implicitamente la funzione 
        % repmat(neuralNetwork.b{l}, size(outputOfPreviousLayer, 1), 1), che
        % forma una nuova matrice che ha lo stesso numero di colonne di
        % b{l}, ma ripete i valori di queste colonne per |NumVettoriInput|
        % righe. 
		neuralNetwork.a{l} = neuralNetwork.a{l} + neuralNetwork.b{l};
		% A questa somma viene applicata la funzione di attivazione
        % del layer corrente.
        neuralNetwork.z{l} = neuralNetwork.g{l}(neuralNetwork.a{l});
        % Aggiorno l'output corrente a quello di questo layer (verra' poi
        % usato nel calcolo del prossimo input, nella prossima iterazione).
        outputOfPreviousLayer = neuralNetwork.z{l};
    end
    
    % Controlla se il flag passato in input riguardo il soft-max e' true. 
    if useSoftmax
        % Applica il soft-max agli output della rete.
        % Il softmax è stato implementato da noi perche' quella di MATLAB aveva un comportamento inaspettato.
        sftmx = exp(neuralNetwork.z{neuralNetwork.numOfHiddenLayers + 1}) ./ sum(exp(neuralNetwork.z{neuralNetwork.numOfHiddenLayers + 1}), 2);
        neuralNetwork.z{neuralNetwork.numOfHiddenLayers + 1} = sftmx;
    end
end