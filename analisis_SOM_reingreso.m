
echo on;
% Este archivo entrena un mapa autoorganizado con la clasificación de reingreso 
% La seleccion del tamaño y otras caracteristicas del mapa se hace por defecto.
% Permite visualizar los mapas generados mediante U-matrix, y las componentes de entrada
% mediante histogramas de activacion de color diferente para cada clase.
echo off;
clear

flag = input('\nSelecciona el tipo de conjunto de datos (0 = original, 1 = equilibrado): ');
if (isempty(flag) || ~flag )
    reingreso; 
    % fichero de datos
else 
    reingreso_balanceado; % fichero de datos
end

echo on;
%
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
%
echo off;

% Mascara incluyendo todas las variables ( 1 = incluir / 0 = suprimir)
% Indice :  1  2  3  4  5  6  7  8  9  10  11  12 13 14 15 16 17 18 19 20
% 21 22 23 24 25 26
mascara = [ 1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];

indices = find((mascara == 1));           % indices de variables NO enmascaradas
data = data( indices , : );               % Extraemos solamente las variables NO enmascaradas
numinp = length( mascara );               % Numero de variables en la mascara
names_variables = names( indices ); % nombres de variables NO enmascaradas
nombre_mapa = 'SOM Reingreso ';
[ numinpnomask , npat ] = size( data );   % numinpnomask = numero de variables NO enmascaradas

%%%%%%%%%%%%%%% GENERA LA STRUCT DE DATOS DEL SOM  %%%%%%%%%%%%%%%%%%%%%%%%%%
% 'name'  dispone el nombre del mapa: SOM Reingreso
% 'names_variables' dispone el nombre de cada variable de entrada al mapa
% 'labels' dispone el nombre para cada patron, en este caso etiquetas de clase
sD = som_data_struct( data' , 'name' , nombre_mapa , 'comp_names' , names_variables , 'labels' , num2str( clase ) );

% Normalizamos cada variable de entrada independientemente en el rango [0 1]
sD = som_normalize( sD , 'range' ); 

% Determinando el tamaño del mapa
% mapsize = [];
mapsize = input('\nIntroduce el tamaño del mapa entre corchetes (preferiblemente, rectangular): ');
lattice = input('\nIntroduce el tipo de mapa (0 = rectangular, 1 = hexagonal): ');
if (isempty(lattice) || lattice)
   lattice = 'hexa';
else 
   lattice = 'rect';
end

%%%%%%%%%%%%%%% ENTRENAR EL SOM  %%%%%%%%%%%%%%%%%%%%%%%%%%
% 'lattice' determina entre mapa hexagonal o rectangular
% 'msize' determina el tamaño del mapa
if (isempty(mapsize))
   sM = som_make(sD, 'lattice', lattice); % SOM entrenado con las opciones por defecto del SOMPACK
else
   sM = som_make(sD, 'msize', mapsize, 'lattice', lattice); % SOM entrenado con el tamaño del mapa introducido
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% RESULTADOS DE LA CLASIFICACION CON EL MAPA ENTRENADO  %%%%%%%%%%%%%%%
%%% SOM_UNIT_CLASSIFICATION etiqueta las neuronas del mapa %%%%%%%%%%%%%%%%%%%%%%%
[ unit_class_index , unit_class_label] = som_unit_classification( sM , sD , clase );
sM.labels = unit_class_label;

%%% SOM_DATA_CLASSIFICATION etiqueta los patrones con las neuronas del mapa
[ data_class_index , data_class_label ] = som_data_classification( sM , sD , unit_class_index );

fprintf('\nResultados de la clasificacion con %d variables no enmascadas:' , sum( mascara ) );
for k = 1:numinp
    if mascara( k )
        fprintf('\n%d - %s' , k , names{ k } );
    end
end
fprintf('\n------------------------------------\n' );

% Calculamos la matriz de confusion y el indice Kappa
[ conf_mat , Kappa ] = confussion_matrix( data_class_index , clase );
fprintf('\nMatriz de confusion:' );
conf_mat    %#ok<NOPTS>
fprintf('\nKappa: %1.10f ' , Kappa  );   

%%%%%%%%%%%%%%%%%%% Representamos el mapa en dos formas diferentes  %%%%%%%%%%%%%%
U = som_umat( sM ); % calcula U-matrix

f1 = figure;  % figura 
set(f1 , 'numbertitle' , 'off');
colormap(1-gray)
subplot( 1 , 3 , 1 ); % pinta la U-Matrix
h = som_cplane( [sM.topol.lattice , 'U' ] , sM.topol.msize , U(:) );  
%set( h , 'Edgecolor' , 'none' );
title( 'U-matrix' );

subplot( 1 , 3 , 2 ); % pinta la D-matrix
Ud = U( 1:2:size( U , 1 ) , 1:2:size( U , 2 ) ); % extrae las componentes impares de la U-Matrix
h = som_cplane( sM , Ud(:) );
%set( h , 'Edgecolor' , 'none'); 
title( 'D-matrix' ); 

% Respuesta difusa calculada sumando 1./(1+(q/a)^2)
% para cada muestra, donde 'q' es un vector que contiene
% la distancia de cada muestra hasta cada prototipo de las unidades del mapa
% y 'a' es el promedio del error de cuantización
subplot( 1 , 3 , 3 ); % pinta la respuesta difusa
hf = som_hits( sM , sD.data , 'fuzzy' ); % Calculamos la respuesta difusa
h = som_cplane( sM , hf );
title( 'Respuesta Difusa' ); 
set(f1 , 'name' , 'U-MATRIX - D-MATRIX - RESPUESTA DIFUSA');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pinta la U-matrix con las activaciones de las muestras de las clases en colores
% Se representan conjuntamente los mapas de la distribución de valores de cada componente de entrada
%       Nº Clase:     Clase:                  Numero de casos:      Color:
%       1             No Reingresa		           439               rojo
%       2             Si Reingresa                  49              amarillo 
%
DATAHITS = struct( 'hits' , [] ); % creamos una struct para guardar las activaciones
muestras = sD.data( 1:numpatclases(1)   , : ); % primera clase
DATAHITS.hits = som_hits( sM , muestras ); % obtenemos los histogramas de activación para la 1 clase
for n = 2:1:numclases
    muestras = sD.data( sum(numpatclases(1:n-1))+1:sum(numpatclases(1:n)) , : );   % muestras de la primera clase 
    DATAHITS( n ).hits = som_hits( sM , muestras ); % obtenemos los histogramas de activación para la clase n
end
clear muestras;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Representamos los histogramas en U-matrices separadas
f2 = figure;  % figura 
set( f2 , 'numbertitle' , 'off');
colormap( 1 - gray );
som_show(sM,'umat',{'all','1-No Reingresa'},'umat',{'all','2-Si Reingresa'},...
    'bar','none')%, 'subplots' , [ 1 3 ])
colores_clases = ['r','y'];
for n = 1:1:numclases
    som_show_add( 'hit' , DATAHITS( n ).hits , 'Subplot' , n , 'Markercolor' , colores_clases( n ) );  % activaciones
end
set(f2 , 'name' , 'ACTIVACION DE LAS CLASES DE Reingreso SOBRE LA U-MATRIX');
set( f2 , 'Position' , [ 90    91   740   587 ] );

% Representamos la U-matrix y las matrices de distancias para cada componente de entrada
colormap(1-gray)
fprintf('\n\nVISUALIZANDO COMPONENTES\n');
cabecera = 'SOM - Componentes Originales: ';

ind = input('Numero de componentes para visualizar en cada ventana (salir -> enter): ');
if ( ~isempty(ind))
    if (ind > 0 && ind < numinpnomask )
        num_figures = floor(numinpnomask/ind); % numero de figuras completas
        last_figure = numinpnomask - num_figures*ind; % ultima figura no completada
        
        for m = 1:num_figures
            H = figure;
            set( H , 'menuBar' , 'none');
            set( H , 'numbertitle' , 'off');
            colormap(1-gray);
            som_show( sM , 'umat' , { 'all' , 'Clases' } , 'comp' , (m - 1)*ind + 1 : m*ind ); % componentes
            set( H , 'name' , [ cabecera  num2str( indices( (m - 1)*ind + 1 : m*ind ) ) ] ); % nombre de figura con indices de componentes originales
            set( H , 'Position' , [ 8   231   667   506 ] );
            % Añadimos los histogramas de activación sobre la U-matrix
            % 1-psoriasis en rojo, 2-seboreic dermatitis en amarillo , 3-lichen planus en verde,
            % 4-pityriasis rosea en cian , 5-cronic dermatitis en azul , 6-pityriasis rubra pilaris en magenta
            for n = 1:1:numclases
                som_show_add( 'hit' , DATAHITS( n ).hits , 'Subplot' , 1 , 'Markercolor' , colores_clases( n ) );  % activaciones
            end
        end
        
        if last_figure
            H = figure;
            set( H , 'menuBar' , 'none');
            set( H , 'numbertitle' , 'off');
            colormap(1-gray);
            som_show( sM , 'umat' , {'all','Clases'} , 'comp' , num_figures*ind + 1 : numinpnomask ); % componentes
            set( H , 'name' , [ cabecera  num2str( indices( num_figures*ind + 1 : numinpnomask ) ) ]); % nombre de figura con indices de componentes originales
            set( H , 'Position' , [ 8   231   667   506 ] );
            % Añadimos los histogramas de activación sobre la U-matrix
           
            for n = 1:1:numclases
                som_show_add( 'hit' , DATAHITS( n ).hits , 'Subplot' , 1 , 'Markercolor' , colores_clases( n ) );  % activaciones
            end
        end
    end
end

% %%% SOM_UNIT_CLASSIFICATION etiqueta las neuronas del mapa %%%%%%%%%%%%%%%%%%%%%%%
% [ unit_class_index , unit_class_label] = som_unit_classification( sM , sD , clase );
% sM.labels = unit_class_label;
% 
% %%% SOM_DATA_CLASSIFICATION etiqueta los patrones con las neuronas del mapa
% [ data_class_index , data_class_label ] = som_data_classification( sM , sD , unit_class_index );
% 
% fprintf('\nResultados de la clasificacion con las variables no enmascadas:');
% 
% % calculamos la matriz de confusion y el indice Kappa
% [ conf_mat , Kappa ] = confussion_matrix( data_class_index , clase ) %#ok<NOPTS>

