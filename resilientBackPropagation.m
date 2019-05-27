function [neuralNetwork, deltaW, deltaB] = resilientBackPropagation(neuralNetwork, derivativeB, derivativeW, derivativeBPrec, derivativeWPrec, deltaWPrec, deltaBPrec, etaMinus, etaPlus, epoch)
% Aggiornamento della rete con resilient back propagation.
% 
% Parametri di input
%   neuralNetwork : Rete neurale istanziata con la funzione newFFMLNeuralNetwork.
%   derivativeB : Array cell che associa ad ogni layer di nodi le derivate
%                 della funzione di errore calcolate per ogni bias.
%   derivativeW : Array cell che associa ad ogni layer di pesi le derivate
%                 della funzione di errore calcoalte per ogni peso.
%   derivativeBPrec :  Array cell che associa ad ogni layer di nodi le derivate 
%                       della funzione di errore calcolate per ogni bias,
%                       calcolate nell'epoca precedente.
%   derivativeWPrec : Array cell che associa ad ogni layer di pesi le derivate
%                     della funzione di errore calcoalte per ogni peso,
%                     calcolate nell'epoca precedente.
%   deltaWPrec : Array cell che associa ad ogni layer di pesi le variazioni
%                effettuate sui pesi nell'epoca precedente.
%   deltaBPrec : Array cell che associa ad ogni layer di nodi le variazioni
%                effettuate sui bias nell'epoca precedente.
%   etaMinus : Numero reale piccolo che rappresenta il fattore moltiplicativo 
%              per gli scostamenti precedenti della matrice dei pesi, quando la 
%              derivata della funzione di errore e' discorde con quella precedente. 
%              Valore consigliato : 0.5.
%   etaPlus : Numero reale piccolo che rappresenta il fattore moltiplicativo rispetto
%             allo scostamento precedente della matrice dei pesi, quando la 
%             derivata della funzione di errore e' concorde con quella precedente. 
%             Valore consigliato : 1.2. 
%   epoch :   Epoca corrente.
%
% Parametri di output
%   neuralNetwork : la rete in input, con pesi e bias aggiornati.
%   deltaW : Array cell che associa ad ogni layer di pesi la variazione
%            effettuata dall'aggiornamento su ogni peso.
%   deltaB : Array cell che associa ad ogni layer di nodi la variazione
%            effettuata dall'aggiornamento su ogni bias.
%
    
    % Per ogni hidden layer e layer di output della rete, aggiorna pesi e
    % bias con uno scostamento.
    for l = 1 : neuralNetwork.numOfHiddenLayers+1
        % Nella prima epoca lo scostamento assume un valore standard.
        if epoch == 1
            % Il delta della prima epoca coincide con quello utilizzato
            % nell'algoritmo standard della discesa del gradiente.
            deltaW{l} = derivativeW{l} * (-0.000001);
            deltaB{l} = derivativeB{l} * (-0.000001);
        
        else
            % Calcolo della concordanza dei segni tra le derivate di pesi e
            % bias calcolate nell'epoca precedente, rispetto a quelli
            % calcolati nell'epoca attuale.
            concordanceW = derivativeWPrec{l} .* derivativeW{l};
            concordanceB = derivativeBPrec{l} .* derivativeB{l};
            
            % Gli scostamenti vengono inizializzati a 0.
            deltaW{l} = zeros(size(deltaWPrec{l},1),size(deltaWPrec{l},2));
            deltaB{l} = zeros(size(deltaBPrec{l},1),size(deltaBPrec{l},2));
            
            % Per i pesi con derivate concordi, gli scostamenti applicati
            % nell'epoca precedente vengono incrementati con il fattore
            % moltiplicativo etaPlus.
            deltaW{l}(concordanceW > 0) = deltaWPrec{l}(concordanceW > 0) * etaPlus;
            % Per i pesi con derivate discordi, gli scostamenti applicati
            % nell'epoca precedente vengono decrementati con il fattore
            % moltiplicativo etaMinus e invertiti di segno.
            deltaW{l}(concordanceW < 0) = deltaWPrec{l}(concordanceW < 0) * (-etaMinus);

            % Per i bias con derivate concordi, gli scostamenti applicati
            % nell'epoca precedente vengono incrementati con il fattore
            % moltiplicativo etaPlus.
            deltaB{l}(concordanceB > 0) = deltaBPrec{l}(concordanceB > 0) * etaPlus;
            % Per i bias con derivate discordi, gli scostamenti applicati
            % nell'epoca precedente vengono decrementati con il fattore
            % moltiplicativo etaMinus e invertiti di segno.
            deltaB{l}(concordanceB < 0) = deltaBPrec{l}(concordanceB < 0) * (-etaMinus);
        end
        
        % I pesi e i bias vengono aggiornati usando gli scostamenti.
        neuralNetwork.b{l} = neuralNetwork.b{l} + deltaB{l};
        neuralNetwork.W{l} = neuralNetwork.W{l} + deltaW{l};
    end
end

