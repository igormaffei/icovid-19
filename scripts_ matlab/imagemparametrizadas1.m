%A=rgb2gray(imread('Documentos/Projeto COVID-19 IME/covid-chestxray-dataset-master/images/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png'));
%A=imresize(A,[582 483]);
%imwrite(A,'Documentos/Projeto COVID-19 IME/Imagens Parametrizadas/imagem1.png');

CY=rgb2gray(imread('Documentos/Projeto COVID-19 IME/covid-chestxray-dataset-master/images/cavitating-pneumonia-4-day28-L.png'));
CY=imresize(CY,[582 483]);
imwrite(CY,'Documentos/Projeto COVID-19 IME/Imagens Parametrizadas/imagem100.png');