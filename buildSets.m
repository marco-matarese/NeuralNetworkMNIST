function[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = buildSets(digits, labels, trainingSetSize, validationSetSize, testSetSize, asBits)
% La funzione buildSets estrae casualmente dal dataset MNIST, precedentemente estratto
% tramite loadMNISTDataset, tre matrici DISGIUNTE di dati distinti: training
% set, validation set e test set. La dimensione di ogni matrice dipende dai
% parametri specificati in input. NOTA BENE: i parametri di input che
% contengono la dimensione di ogni matrice estratta dovranno essere divisibili 
% per 10, altrimenti verra' lanciato un errore dalla funzione, che non 
% ritornera' nulla. Inoltre, la label setLabel(i) corrisponde all'immagine
% setData(i, :), per costruzione, per ogni set.
%
% Parametri di input
%   digits : matrice di immagini di MNIST 60000x784, precedentemente ritornata 
%            dalla funzione loadMNISTDataset.
%   labels : matrice di labels di MNIST 60000x1, precedentemente ritornata
%            dalla funzione loadMNISTDataset.
%   trainingSetSize : specifica quanti dati vogliamo estrarre da MNIST per
%                     il training set. Deve essere un valore divisibile per
%                     10.
%   validationSetSize : specifica quanti dati vogliamo estrarre da MNIST per
%                       il validation set. Deve essere un valore divisibile per
%                       10.
%   testSetSize : specifica quanti dati vogliamo estrarre da MNIST per
%                 il test set. Deve essere un valore divisibile per
%                 10.
%   asBits : se contiene il valore true, l'output dei setLabels verra'
%            rappresentato come una matrice sizex10. Se una riga e' riferita
%            al digit i, allora solo il bit i-simo della riga sara' alto. Se
%            contiene il valore false, l'output dei set verra' rappresentato
%            come una matrice sizex1 che conterra' il label originale per
%            ogni elemento estratto.
%
% Parametri di output
%   trainingSetData : e' una matrice trainingSetSizex784 che contiene il
%                     sottoinsieme di immagini distinte casuali estratte da digits.
%   trainingSetLabels : e' una matrice trainingSetSizex1 che contiene il
%                       sottoinsieme di labels distinte casuali estratte da labels.
%   validationSetData : e' una matrice validationSetSizex784 che contiene il
%                       sottoinsieme di immagini casuali distinte estratte
%                       da digits che non sono state precedentemente
%                       inserite in trainingSetData.
%   validationSetLabels : e' una matrice validationSetSizex1 che contiene il
%                         sottoinsieme di labels casuali distinte estratte 
%                         da labels che non sono state precedentemente
%                         inserite in trainingSetLabels.
%   testSetData : e' una matrice testSetSizex784 che contiene il
%                 sottoinsieme di immagini casuali distinte estratte da digits
%                 che non sono state precedentemente inserite in trainingSetData
%                 e in validationSetData.
%   testSetLabels : e' una matrice testSetSizex1 che contiene il
%                   sottoinsieme di labels casuali distinte estratte da labels che 
%                   non sono state precedentemente inserite in
%                   trainingSetLabels e in validationSetLabels.

    % Controllo se le varie cardinalita' sono valori divisibili per 10
    % Se non lo sono, la funzione termina immediatamente con un errore
    if (mod(trainingSetSize, 10) ~= 0) || (mod(validationSetSize, 10) ~= 0) || (mod(testSetSize, 10) ~= 0)
        error('setSize parameters must be a number that can be divided by 10.');
        return
    end
    
    % Array contenente gli indici di immagini del dataset MNIST che sono
    % stati generati gia' in precedenza (garantisce che ogni elemento nei
    % set sia disgiunto e distinto)
    indexAlreadyTaken = zeros(1, (trainingSetSize+validationSetSize+testSetSize));
    % Contatore che ricorda l'ultima posizione di indexAlreadyTaken
    % in cui ho inserito un valore.
    lastPosition = 1;
    
    % Calcolo la matrice delle immagini e l'array delle lables per il
    % training set. Inoltre, ritorno l'array indexAlreadyTaken aggiornato
    % con gli indici delle immagini/labels di MNIST che ho inserito nel 
    % training set
    [trainingSetData, trainingSetLabels, indexAlreadyTaken, lastPosition] = buildSet(digits, labels, trainingSetSize, indexAlreadyTaken, lastPosition, asBits);
    
    % Calcolo la matrice delle immagini e l'array delle lables per il
    % validation set. Inoltre, ritorno l'array indexAlreadyTaken aggiornato
    % con gli indici delle immagini/labels di MNIST che ho inserito nel 
    % validation set
    [validationSetData, validationSetLabels, indexAlreadyTaken, lastPosition] = buildSet(digits, labels, validationSetSize, indexAlreadyTaken, lastPosition, asBits);
    
    % Calcolo la matrice delle immagini e l'array delle lables per il 
    % test set
    [testSetData, testSetLabels] = buildSet(digits, labels, testSetSize, indexAlreadyTaken, lastPosition, asBits);
end

function [setData, setLabels, indexAlreadyTaken, lastPosition] = buildSet(digits, labels, setSize, indexAlreadyTaken, lastPosition, asBits)

    % Contatore per incrementare gli indici della matrice delle immagini e
    % l'array delle labels da estrarre dal dataset
    j = 1;
    % Matrice delle immagini
    setData = zeros(setSize, 784);
    % Array delle labels
    if asBits
        setLabels = zeros(setSize, 10);
    else
        setLabels = zeros(1, setSize);
    end
    
    % Numero di immagini da prendere per ogni cifra
    numOfDigits = setSize/10;
    % Array che contiene nella cella i-sima il numero di cifre inserite
    % nella matrice delle immagini per la cifra i
    digitsCounter = zeros(1, 10);
    % Cifra da cui si parte
    currentDigit = 0;
    
    % Finche' non ho recuperato tutte le cifre (da 0 a 9)
    while currentDigit <= 9
        % Finche' non ho inserito, per la cifra corrente, numOfDigits
        % valori distinti nella matrice delle immagini
        while digitsCounter(currentDigit+1) <= numOfDigits-1
            % Genera un numero casuale da 1 a 60000
            randomIndex = floor((60000-1).*rand(1) + 1);
            % Se questo numero e' l'indice nella matrice del dataset la
            % cui cifra rappresentata e' la cifra corrente, e se non ho gia'
            % inserito questa specifica immagine nella matrice di output
            if (labels(randomIndex) == currentDigit) && (~ismember(randomIndex, indexAlreadyTaken))
                % Ho trovato una nuova cifra valida
                digitsCounter(currentDigit+1) = digitsCounter(currentDigit+1) + 1;
                % Aggiungi questa label all'array delle label
                if asBits
                    setLabels(j, currentDigit+1) = 1;
                else
                    setLabels(j) = currentDigit;
                end
                % Aggiungi questa immagine alla matrice di output
                setData(j, :) = digits(randomIndex, :);
                % Aggiungi questo indice a quelli gia' inseriti
                indexAlreadyTaken(lastPosition) = randomIndex;
                % Prepara la prossima locazione per la matrice di output e
                % per l'array delle label di output
                j = j + 1;
                % Prepara la prossima posizione dell'array che mantiene gli
                % indici gia' inseriti
                lastPosition = lastPosition + 1;
            end
        end
        % Passa alla prossima cifra
        currentDigit = currentDigit + 1;
    end
end