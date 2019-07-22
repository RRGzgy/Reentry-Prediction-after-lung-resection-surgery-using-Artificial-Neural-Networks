
clc
%
clear;
flag = input('\nSelecciona el tipo de conjunto de datos (0 = original, 1 = equilibrado): ');
if (isempty(flag) || ~flag )
    reingreso;          % fichero de datos
else 
    reingreso_balanceado; % fichero de datos
end

%  Informacion sobre las variables:
%
%      1: Hombre
%      2: Mujer
%      3: Edad
%      4: ASA
%      5: Hipertension
%      6: Vascularizacion Arterial Periferica
%      7: Insuficiencia Renal
%      8: Cardiopatia Isquemica
%      9: Insuficiencia Cardiaca
%     10: Arritmia
%     11: Ictus
%     12: Cirugia Previa
%     13: Tabaco
%     14: SRS
%     15: Estancia Prolongada
%     16: Reingreso UCI/REA
%     17: IOT Prolongada
%     18: REIOT mismo ingreso
%     19: Numero de Comorbilidades
%     20: Numero de Complicaciones
%     21: Comorbilidades
%     22: Complicaciones
%     23: VATS
%     24: Toracotomia
%     25: EPOC
%     26: DINDO
%     
%  Clases:
%       
%       Codigo Clase:   Clase:                  Numero de casos:
%       1             No reingresa			        439
%       2             Si Reingresa                   49

% Mascara incluyendo todas las variables ( 1 = incluir / 0 = suprimir)
% Indice :  1  2  3  4  5  6  7  8  9  10  11  12 13 14 15 16 17 18 19 20
% 21 22 23 24 25 26
mascara = [ 1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];

indices = 1:1:ndata;                % vector de indices de variables 
indices = indices((mascara == 1));  % indices de variables NO enmascaradas
data = data( indices , : );         % Extraemos solamente las variables NO enmascaradas
names_variables = names( indices ); % nombres de variables NO enmascaradas
[ numinp , npat ] = size( data );   % numinp = numero de variables No enmascaradas, npat = numero de ejemplos

[ lx , ly ] = size( clases ); % Clases es una matriz dispersa
out = full( clases ); % La convertimos a una matriz completa

[ dataesc , minp , maxp ] = premnmx( data );
fprintf( sprintf( '%s%d' , '\nDimensión del espacio de proyección de PCA: ' , numinp ));
odim = numinp;
[ datapca , V , me , autovalues ] = pcaproj( dataesc' , odim );
datapca = datapca';
[ dim , npat ] = size( datapca );
fprintf('\nValores propios de la matriz de covarianza:');
autovalues' %#ok<*NOPTS>

% Dibujamos los datos en el subespacio proyectado
fprintf('\nVISUALIZANDO PARES DE COMPONENTES\n');
while (1)
   ind = input('Dos índices de componentes para visualizar entre [] (para salir -> enter): ');
   if isempty(ind); break; end;
   H = drawkernel( [ ind(1) ind(2) ] , datapca , out );
end
if (flag)
    fprintf('\nVISUALIZANDO RELACIÓN DE COMPONENTES PCA Y LAS EQUILIBRADAS\n');
else
    fprintf('\nVISUALIZANDO RELACIÓN DE COMPONENTES PCA Y LAS ORIGINALES\n');
end
ind = input('Indices de componentes para visualizar (con []): ');
if isempty(ind); return; end;

% Dibujamos la relación entre los componentes de PCA y las entradas
figure;
lx = length( ind );
x = 1:1:numinp;
for j = 1:lx, % pintamos las graficas horizontales
   H = subplot( lx , 1 , j );
   bar( x , V( ind( j ) , : ));
   if j == 1;
       if (flag)
           title('RELACIÓN ENTRE COMPONENTES PCA Y DATOS DE ENTRADA EQUILIBRADOS');
       else
           title('RELACIÓN ENTRE COMPONENTES PCA Y DATOS DE ENTRADA ORIGINALES');
       end
   end
   axis tight;
   if j ~= lx
      set( H , 'XTick' , x );
      set( H , 'XTickLabel' , [] );
  else
      set( H , 'XTick' , x );
      set( H , 'XTickLabel' , indices );
   end
   ylabel(sprintf('%s%d' , 'PCA' , ind( j )));
   grid;
end

if (flag)
    xlabel('COMPONENTES DE ENTRADA DEL PROBLEMA EQUILIBRADO');
else
    xlabel('COMPONENTES DE ENTRADA DEL PROBLEMA ORIGINAL');
end

