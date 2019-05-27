function neuralNetwork = backPropagation(neuralNetwork, T, E)
% Calcola i delta per ogni layer, propagando la rete
% all'indietro (a partire dal nodo di output verso il nodo di input).
%
% Parametri di input
%   neuralNetwork : Rete neurale ritornata dalla funzione forwardPropagation.
%   T : Matrice di valori tale che la riga i-sima rappresenta il target
%       da ottenere rispetto ai valori di output generati dalla rete neurale.
%   E : Puntatore alla funzione da utilizzare per il calcolo dell'errore da utilizzare.
%
% Parametri di output
%   delta : Array cell di matrici tale che ogni riga j dell'i-sima matrice rappresenta le
%           derivate parziali della funzione di errore rispetto ai pesi del layer i 
%           per il vettore di input j (le colonne sono gli input dei singoli nodi).
    
    % A partire dal layer di output, calcolo il rispettivo delta. Il delta
    % per il layer di output si calcola moltiplicando punto-punto la derivata prima della
    % funzione di attivazione del layer corrente valutata nell'input del nodo di output e la
    % derivata parziale della funzione di errore rispetto i valori di
    % output del layer di output. Cio' che si ottiene sara' una matrice di
    % dimensione |numElementiDiT|x|numNodiOutput|, cioe' ci sara' una
    % riga per ogni vettore target passato alla funzione (che coincide con
    % il numero di vettori passati in input alla forward propagation) in
    % cui ogni colonna rappresenta la derivata parziale della funzione di
    % errore rispetto al proprio valore di output.
    outputLayer = neuralNetwork.numOfHiddenLayers+1;
    neuralNetwork.delta{outputLayer} = neuralNetwork.outputActivation(neuralNetwork.a{outputLayer}, true) .* E(neuralNetwork.z{outputLayer}, T, true);
    
    % Si procede con il calcolo dei delta per tutti gli hidden layers, a
    % partire dal layer di output.
    for l = outputLayer-1 : -1 : 1
        % Il calcolo del delta per il layer corrente si calcola
        % moltiplicando punto-punto la derivata prima della funzione di
        % attivazione valutata nell'input del nodo ed il prodotto
        % matrice-matrice tra il delta del layer successivo ed i pesi del
        % layer successivo. Si noti che il delta del layer successivo ha
        % dimensione |numRigheDiT|x|numNodiLayerSucc| e la matrice dei
        % pesi del layer successivo ha dimensione
        % |numNodiLayerSucc|x|numNodiLayerCurr|, e quindi la matrice
        % risultante avra' dimensione |numRigheDiT|x|numNodiLayerCurr|.
        % Infine, il prodotto punto-punto si fa con questa matrice
        % risultante e la matrice dei valori di input valutati con la
        % funzione di attivazione corrente, avente dimensione pari a 
        % |numRigheDiT|x|numNodiLayerCurr|, ed ha quindi senso
        % effettuare il prodotto punto-punto per generare la matrice finale
        % di dimensione |numRigheDiT|x|numNodiLayerCurr|.
        neuralNetwork.delta{l} = neuralNetwork.g{l}(neuralNetwork.a{l}, true) .* (neuralNetwork.delta{l+1} * neuralNetwork.W{l+1});
    end
end
