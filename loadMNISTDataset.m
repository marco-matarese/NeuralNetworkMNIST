function [digits, labels] = loadMNISTDataset(imagesFilename, labelsFilename)
% Il dataset MNIST contiene 60.000 digits di dimensione 28x28 pixel. Questa
% procedura richiama due funzioni realizzate dall'universita' di Stanford che
% permettono di estrarre, dai dataset MNIST specificati nei parametri della
% funzione delle immagini e delle labels, due matrici: digits e labels.
% 
% Parametri di input
%   imagesFilename : path al file contenente il dataset delle immagini
%   labelsFilename: path al file contenente il dataset delle labels
%
% Parametri di output
%   digits : e' una matrice di dimensione 60000x784. Sulle righe della
%            matrice sono riportati i digits e sulle righe le
%            rappresentazioni dei digits. Ad esempio, digits(i,:) e' la
%            rappresentazione del digit i-simo. Per strutturare un digit
%            secondo la sua rappresentazione originale basta riorganizzare 
%            la matrice digits(i,:) come una matrice 28x28, utilizzando
%            la funzione reshape(digits(i,:), [28,28]).
%   labels : e' una matrice 60000x1. Ogni riga della matrice contiene la
%            label per la digit corrispondente. Ad esempio, labels(i) e'
%            l'etichetta che descrive il contenuto del digit i-simo.
    digits = (loadMNISTImages(imagesFilename))';
    labels = loadMNISTLabels(labelsFilename);
end

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end