
clear;
warning off

% Leemos los datos de entrada y salidas
reingreso_balanceado;
N_OUT = 2; % numero de salidas del problema

% Mascara con todas las entradas 
% Indice :  1  2  3  4  5  6  7  8  9  10  11  12 13 14 15 16 17 18 19 20
% 21 22 23 24 25 26
mascara = [ 1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];


indices = find( ( mascara == 1 ) );       % extraemos los indices de variables NO enmascaradas
data = data( indices , : );               % extraemos solamente las variables NO enmascaradas
numinp = sum( mascara );                  % numinp = numero de variables NO enmascaradas
names_variables = names( indices );       % nombres de variables NO enmascaradas

[ nclases , num_pat ] = size( clases ); % clases es una matriz dispersa
outclase = full( clases );           % outclase es una matriz completa
% Calculamos las prob a priori de las clases
prob_priori = numpatclases / num_pat; 

% Numero de salidas de la red
fprintf('\nÍNDICES DE SALIDAS PARA ENTRENAR LAS REDES' );
ind_salidas = input('\n(por defecto [1 2]): ');
if ( isempty( ind_salidas ) ); 
    ind_salidas = ( 1:1:N_OUT ); % por defecto [ 1 2 ]
end
out = outclase( ind_salidas , : ); % seleccionamos las respuestas deseadas para las salidas que tenga la red
[ num_out dummy ] = size( out );   % Numero de clases de salida
ind_out = 1:1:num_out;             % vector de indices de salidas
clase = ind_out * out;             % matriz de clasificacion de patrones no-dispersa
prob_priori = prob_priori( ind_salidas );% seleccionamos las prob a priori de las clases seleccionadas

% Valor del umbral con el que comparar las salidas de la red
umbral = input('\nVALOR DE UMBRAL DE SALIDA DE LA RED (por defecto 0.1): ');
if ( isempty( umbral ) || ( umbral < 0 ) ); 
    umbral = 0.1; % umbral selecciona una respuesta como no nula (alguna de las salidas de la red lo supera)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%               INTRODUCIENDO DATOS DE ENTRENAMIENTO                       %%%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unidades Ocultas en las diferentes arquitecturas
nhunit = input('\nNÚMERO DE UNIDADES OCULTAS (por defecto [ 0 5 8 10 12 15 20 ]): ');
if ( isempty( nhunit ) ); 
    nhunit = [ 0 5 8 10 12 15 20 ]; 
end

n_arch = length(nhunit); % numero de arquitecturas
n_pool = input('\nNÚMERO DE REPETICIONES DE CADA RED (por defecto 20): ');
if ( isempty( n_pool ) || n_pool <= 0 ); 
    n_pool = 20; 
end

% Metodo de entrenamiento
% funtrain = 'trainlm'; % por defecto L-M

% preguntamos el método de entrenamiento
funtrain = pidefuntrain;
if isempty( funtrain ); 
    funtrain = 'trainlm'; % por defecto L-M
end 
% Número máximo de ciclos de entrenamiento
% (disminuirlo para los metodos cuasi-Newton y de grad.conjugados)
numciclos = input('\nNÚMERO MÁXIMO DE CICLOS DE ENTRENAMIENTO (por defecto 100): ');
if isempty( numciclos ); numciclos = 100; end 

fprintf('\nEntrenamiento con %d variables no enmascadas:' , numinp );
%fprintf('\n------------- CLÍNICAS -------------' );
for k = 1:length( mascara )
    if mascara( k )
        fprintf('\n%d - %s' , k , names{ k } );
    end
%     if k == VAR_CLINICAS
%         fprintf('\n--------- HISTOPATOLÓGICAS ---------' );
    end
% end
% fprintf('\n------------------------------------\n' );

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%             AQUI SE ENTRENAN Y EVALUAN n_eval REDES                      %%%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Toterr = zeros( n_pool, n_arch );  % matriz con el MSE en el conjunto de datos: n_pool x n_arch
Totpsnr = zeros( n_pool, n_arch );  % matriz con el psnr en el conjunto de datos: n_pool x n_arch
Totvar = zeros( n_pool, n_arch );  % matriz con los desv.standar del MSE en el conjunto de datos: n_pool x n_arch
Kappa_arq = zeros( 1, n_arch );  % Kappa medio obtenido de las redes del pool en cada arquitectura
min_error = zeros( 1, n_arch ); % minimo MSE de las redes del pool en cada arquitectura
max_error = zeros( 1, n_arch ); % maximo MSE de las redes del pool en cada arquitectura
min_psnr = zeros( 1, n_arch ); % minimo PSNR de las redes del pool en cada arquitectura
max_psnr = zeros( 1, n_arch ); % maximo PSNR de las redes del pool en cada arquitectura
Toterr2 = zeros( 1, n_arch ); % promedio de los MSE de todas las redes del pool
Totvar2 = zeros( 1, n_arch ); % promedio de las desv.standar del MSE de todas las redes del pool
Totpsnr2 = zeros( 1, n_arch ); % promedio de los PSNR de todas las redes del pool
Totpsnrvar2 = zeros( 1, n_arch ); % desv.standar de los PSNR de todas las redes del pool
nhunit_arqmin = nhunit( 1 ); % numero unidades en la arquitectura de red con minimo error, inicialmente la primera
indpoolmin = 1; % indice de la red del POOL con minimo error, inicialmente la primera
minerrortotal = num_pat * num_out ; % error inicial alto igual al numero de muestras x salidas
conf_mat_arq = []; % 3D para guardar matrices de confusion de las salidas promedio en el POOL de cada arquitectura
warning off MATLAB:nearlySingularMatrix;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      CREANDO Y SIMULANDO CADA RED  num_arquitecturas x num_pool           %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:n_arch
    % Creamos la red MLP nm-esima con una capa oculta con nhunit(i) bipolares 
    % y una capa de salida con 1 unidad unipolar
    datam = [ min( data, [], 2 ) , max( data, [], 2 ) ];
    Ymed = zeros( num_out , num_pat ); % matriz de salidas promedio del pool de redes
    base_psnr = 10*log10( num_pat * num_out ); % termino constante del calculo de PSNR
    for j = 1:n_pool
        if ( nhunit( i ) ) % distinguimos red de dos capas de la red de una capa
            net = newff( datam , [ nhunit( i ) num_out ], { 'tansig' , 'logsig' } , funtrain ); % dos capas
        else
            net = newff( datam , num_out , { 'logsig' }, funtrain ); % una capa
        end
        net.trainParam.epochs = numciclos;
        % MSE buscado
        net.trainParam.goal = 1e-4;
        % Entrenamos a la Red
        fprintf('\nENTRENANDO LA RED %d EN LA ARQUITECTURA CON %d NEURONAS OCULTAS:' , j , nhunit( i ) );
        net = train( net , data , out );      
        % Obtenemos la respuesta para los patrones
        Y = sim( net , data ); 
        Ymed = Ymed + Y; % acumulamos las salidas
        % Calculamos el vector de errores en cada patron frente a la respuesta deseada
        err = sum( ( ( Y - out ) .^ 2 ) , 1 ); % sumamos el error en todas las salidas
        psnr = base_psnr - 10*log10( sum( err ) ); % PSNR del conjunto de datos x salidas
        % Guardamos los datos de la red en un vector de redes y los errores de las redes
        if ( j == 1 ) % si j==1 creamos el vector de redes y de errores
            mlpnet = { net };
            errnet = err;
        else % en caso contrario, concatenamos las redes y errores
            mlpnet = cat( 1 , mlpnet , { net } );
            errnet = cat( 1 , errnet , err );  % guardamos los vectores de error
        end
        errnetmedio = mean( err );        % error medio en todos los patrones 
        fprintf( '\nMSE medio en la arquitectura: %f\n' , errnetmedio );
        errnervar = var( err );           % varianza del error en todos los patrones
        Toterr( j , i ) = errnetmedio;    % guardamos el error medio 
        Totvar( j , i ) = errnervar;      % guardamos la varianza
        Totpsnr( j , i ) =  psnr;         % guardamos el PSNR de la red
        if ( errnetmedio < minerrortotal )
            nhunit_arqmin = nhunit( i );  % numero unidades en la arquitectura de red con minimo error
            indpoolmin = j;               % indice de la red del pool con minimo error
            minerrortotal = errnetmedio ; % nuevo error minimo
            netmin = net;                 % estructura de red de error minimo
        end
    end  % fin de bucle de j para repeticiones de la misma arquitectura de red
    
    % Buscamos una matriz 'clasificacion' con 1 y 0, donde el 1 indica
    % las salidas de red maximas
    Ymed = Ymed / n_pool; % promediamos las salidas de la red en el pool de redes
    
    % Obtenemos una matriz 'clasificacion' de 1 y 0 para las salidas seleccionadas como
    % maximas que esten por encima del umbral, si no seran todas cero 
    clasificacion = clasificacion_MLP( Y , prob_priori' , umbral );
    
    % Calculamos la matriz de confusion. Si una columna de
    % clasificacion es todo ceros, aparece una clase de rechazo
    [ conf_mat , Kappa ] = confussion_matrix( ind_out * clasificacion , clase );
    
    % Guardamos el vector de redes en una matriz de redes y las mat. conf. promedios
    if ( i == 1 )
        mlpnet2 = mlpnet;
        conf_mat_arq = conf_mat;
    else
        mlpnet2 = cat( 2 , mlpnet2 , mlpnet );
        conf_mat_arq = cat( 3 , conf_mat_arq , conf_mat );
    end
    Kappa_arq( i ) = Kappa;  % guardamos el Kappa medio obtenido de las redes del pool en esta arquitectura
    min_error( i ) = min( Toterr( : , i) ); % guardamos el minimo MSE de las redes del pool en esta arquitectura
    max_error( i ) = max( Toterr( : , i) ); % guardamos el maximo MSE de las redes del pool en esta arquitectura
    min_psnr( i ) = min( Totpsnr( : , i) ); % guardamos el minimo PSNR de las redes del pool en esta arquitectura
    max_psnr( i ) = max( Totpsnr( : , i) ); % guardamos el maximo PSNR de las redes del pool en esta arquitectura    
    Toterr2( i ) = mean( Toterr( : , i) ); % promediamos los MSE de todas las redes del pool
    Totvar2( i ) = mean( Totvar( : , i) ); % promediamos las varianzas de MSE DE todas las redes del pool
    Totpsnr2( i ) = mean( Totpsnr( : , i) ); % promediamos los PSNR de todas las redes del pool
    Totpsnrvar2( i ) = std( Totpsnr( : , i) ); % desv.standar de los PSNR de todas las redes del pool
end % fin del bucle i para nhunit

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           distribucion del MSE en el POOL para cada arquitectura              %%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ftot2 = figure; % figura de Error total de cada arquitectura
set( ftot2 , 'numbertitle' , 'off' );
set( ftot2 , 'name' , 'ERRORES TOTALES EN LAS DIFERENTES ARQUITECTURAS' );
set( ftot2 , 'NumberTitle' , 'off' );
set( ftot2 , 'Name' , 'ESTADÍSTICAS DEL MSE EN CADA ARQUITECTURA' );
H = subplot( 1 , 1 , 1 ); % Handle de los ejes
puntos_linea = 1:1:n_arch;
line( puntos_linea , Toterr2 , 'Marker' , 'o' , 'Color' , 'k' );
hold on;
plot( max_error , 'r*' );
plot( min_error , 'm*');
errorbar( puntos_linea , Toterr2 , Totvar2 )
title('Error Cuadrático Total');
legend( 'Medias'  , 'Máximos' , 'Mínimos' , 'Media +/- DesvStd' );
set( H , 'XTick', puntos_linea );
set( H , 'XTickLabel' , nhunit );
xlabel('Número de unidades ocultas');
ylabel( 'Estadísticas del MSE' );
if (n_arch > 1)
    base_value = min ( [ min( min_error ) , min( Toterr2 - Totvar2 ) ] );
    top_value = max( [ max( max_error ) , max( Toterr2 + Totvar2 ) ] );
    axis( [ 1 , n_arch , base_value , top_value ] );
end
hold off;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           distribucion del PSNR en el POOL para cada arquitectura             %%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ftot3 = figure; % figura de psnr total de cada arquitectura
set( ftot3 , 'numbertitle' , 'off' );
set( ftot3 , 'name' , 'DISTRIBUCIONES DEL PSNR EN LAS DIFERENTES ARQUITECTURAS' );
set( ftot3 , 'NumberTitle' , 'off' );
set( ftot3 , 'Name' , 'ESTADÍSTICAS DEL PSNR EN CADA ARQUITECTURA' );
H = subplot( 1 , 1 , 1 ); % Handle de los ejes
puntos_linea = 1:1:n_arch;
line( puntos_linea , Totpsnr2 , 'Marker' , 'o' , 'Color' , 'k' );
hold on;
plot( max_psnr , 'r*' );
plot( min_psnr , 'm*');
errorbar( puntos_linea , Totpsnr2 , Totpsnrvar2 );
title('PSNR en las arquitecturas');
legend( 'Medias'  , 'Máximos' , 'Mínimos' , 'Media +/- DesvStd' );
set( H , 'XTick', puntos_linea );
set( H , 'XTickLabel' , nhunit );
xlabel('Número de unidades ocultas');
ylabel( 'PSNR( dB )' );
if (n_arch > 1)
    base_value = min ( [ min( min_psnr ) - 10 , min( Totpsnr2 - Totpsnrvar2 - 10 ) ] );
    top_value = max( [ max( max_psnr ) + 10 , max( Totpsnr2 + Totpsnrvar2 + 10 ) ] );
    if base_value < 0; base_value = 0; end
    axis( [ 1 , n_arch , base_value , top_value ] );
end
hold off;

%%
% Escribe los resultados error y las matrices de confusion para
% el promedio de redes del pool de cada arquitectura
for i = 1:n_arch
    fprintf('\nARQUITECTURA CON %i NEURONAS OCULTAS:' , nhunit(i) );
    fprintf('\nResultados para el promedio de las %i redes del POOL:' , n_pool );
    fprintf('\nKappa medio: %7.6f ;' , Kappa_arq( i ) );
    fprintf('\nPSNR medio: %10.6f dB;' , Totpsnr2( i ) );
    muestras_nulas = num_pat - sum( sum( conf_mat_arq( : , : , i ) ) );
    fprintf('\nNúmero de muestras nulas: %i ;' , muestras_nulas );
    fprintf('\nMatriz de confusión del promedio de respuestas del POOL:\n' );
    [ num_rows , num_cols ] =  size( conf_mat_arq( : , : , i ) );
    for p = 1:num_rows
        fprintf(' %5.0f  ' , conf_mat_arq( p , : , i ) );
        fprintf('\n');
    end
end % fin del bucle i para nhunit

%%
% Mostramos los PSNR en las redes de cada arquitectura (n_pool x n_arch)
fprintf('\n-------------------------------------------------------------------------\n' );
fprintf('\nPSNR de las %i redes de cada pool en las %i arquitecturas:\n' ,  n_pool , n_arch );
for i = 1:n_pool
    fprintf(' %6.2f  ' , Totpsnr( i, : ) );
    fprintf('\n');
end

%%
% Obtenemos la respuesta de la red de error minimo y su matriz de confusion
fprintf('\n-------------------------------------------------------------------------\n' );
fprintf('\nRed con error mínimo en arquitectura con %i neuronas ocultas y prueba %i.' , nhunit_arqmin , indpoolmin );
Y = sim( netmin , data ); 
clasificacion = round( Y ); % buscamos el entero mas proximo a la respuesta de salida
maxout = max( Y , [] , 1); % deducimos la salida maxima de la red en cada patron
maxout = repmat( maxout , num_out , 1 );
clasificacion = ( Y == maxout ) .* clasificacion; % podria darse un empate, por lo que esta matriz daria error
[ Matriz_Confusion , Kappa ] = confussion_matrix( ind_out * clasificacion , clase );
muestras_nulas = num_pat - sum( sum( Matriz_Confusion ) );

fprintf('\nARQUITECTURA CON %i NEURONAS OCULTAS:' , nhunit_arqmin );
fprintf('\nKappa: %7.6f ;' , Kappa );
fprintf('\nNúmero de muestras nulas: %i ;' , muestras_nulas );
fprintf('\nMatriz de confusión:\n' );
[ num_rows , num_cols ] =  size( Matriz_Confusion );
for p = 1:num_rows
    fprintf(' %5.0f  ' , Matriz_Confusion( p , : ) );
    fprintf('\n');
end
fprintf('\n-------------------------------------------------------------------------\n' );

%%
% Pintamos en grafico de barras las salidas de la red de menor error para
% ver si hay patrones que puedan estar en varias clases
% pintamos las respuestas en grafico de barras
base = '1:439 No Reingresan - 440:880 Si Reingresan';
xticks = [1 439 880 ];
titulo = '';
barcompout(Y , 'Out_' , base , xticks , titulo );
set( gcf , 'menu' , 'figure' );
set( gcf , 'Name' , sprintf( 'MEJOR RED EN CLASIFICACION CON LAS CLASES: [ %s ]' , num2str( ind_salidas ) ) );

% Guardamos los resultados
save MLP_ARQ_reingreso.mat
