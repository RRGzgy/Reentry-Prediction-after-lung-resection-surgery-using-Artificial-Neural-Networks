% Este fichero lee las redes de tipo MLP entrenadas
% efectuando un proceso de validacion cruzada (CV, Cross Validation)
% con la funcion MLP_CV

clc
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      CARGAMOS LOS DATOS DE LAS REDES GUARDADOS CON LA FUNCION MLP_CV.M     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load MLP_CV_reingreso.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%        SALIDA POR PANTALLA DE LOS DATOS DE CLASIFICACION AL DETALLE       %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\nRESULTADO DE LAS DIFERENTES ARQUITECTURAS:\n' );
fprintf('\nNumero de arquitecturas de redes simuladas: %d' , n_arch );
fprintf('\n  - Neuronas en capa oculta: %d; ' , nhunit );
fprintf('\nUmbral para considerar salida nula en la red: %f' , umbral );
fprintf('\nNumero de Validaciones Cruzadas: %d' , n_eval );
fprintf('\nNumero de repeticiones de cada red (pool): %d' , n_pool );
fprintf('\nEntrenadas con la funcion: %s; durante %d ciclos de entrenamiento' , funtrain , numciclos );
fprintf('\nUtilizando %d variables no enmascadas:\n' , sum( mascara ) );
for k = 1:numinp
    fprintf('%d - %s\n' , indices( k ) , names_variables{ k } );
end
fprintf('\nUtilizando %d salidas de clases:\n' , length( ind_salidas ) );
fprintf('\n  - Clases de la capa de salida: %s; ' , name_clases{ ind_salidas } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%        Error Total en las evaluaciones cruzadas para cada arquitectura        %%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ftot2 = figure; % figura de Error total de cada arquitectura
set( ftot2 , 'numbertitle' , 'off' );
set( ftot2 , 'Name' , 'ESTADISTICAS DEL ERROR CUADRATICO MEDIO DE LAS CV EN CADA ARQUITECTURA' );
H = subplot( 1 , 1 , 1 ); % Handle de los ejes
puntos_linea = 1:1:n_arch;
bar( Toterr_arq , 'g');
hold on;
plot( max_error , 'r*' );
plot( min_error , 'm*');
errorbar( puntos_linea , Toterr_arq , Totstd_arq  , 'ob' );
legend(  'Medias' , 'Maximos' , 'Minimos' , 'Medias +/- DesvStd' );
set( H , 'XTick', puntos_linea );
set( H , 'XTickLabel' , nhunit );
xlabel('Numero de unidades ocultas');
ylabel( 'Estadisticas del error cuadratico medio de las CV' );
axis auto;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%           Kappas en las evaluaciones cruzadas para cada arquitectura          %%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toterr = [];
% conf_mat_arq = []; % matriz 4D para guardar las matrices de confusion promedio de cada arquitectura y evaluacion
% Toterr_arq = zeros( n_arch ); % promedio de errores de todas las evaluaciones 
% Totstd_arq = zeros( n_arch ); % desviacion estandar de todas las evaluaciones
% min_error = zeros( n_arch );  % minimo error de las redes del pool en esta arquitectura
% max_error = zeros( n_arch );  % maximo error de las redes del pool en esta evaluacion
% conf_mat_eval = []; % matriz 3D para guardar las matrices de confusion promedio de cada evaluacion
% Kappa_eval = zeros( 1, n_eval ); % vector para guardar kappas de cada validacion
% Toterr_pool = []; % promedio del error de todas las muestras en esta evaluacion
% Totstd_pool = []; % desviacion estandar de error de los errores de esta evaluacion

Kappa_arq_eval = zeros( n_eval , n_arch ); % Kappas de cada validación en las arquitecturas
Kappa_arq = zeros( 1 , n_arch ); % Kappas del promedio de matrices de confusion en la arquitectura

for i = 1:n_arch    
    for n = 1: n_eval
        Kappa_arq_eval( n , i ) = kappa_index( conf_mat_arq( : , : , n , i ) );  % kappa de la evaluacion y arquitectura
    end
    Matriz_Confusion = sum( conf_mat_arq( : , : , : , i ) , 3 );  % matriz de confusion media en el pool
    Kappa_arq( i ) = kappa_index( Matriz_Confusion );  % kappa de la arquitectura
end % fin del bucle i para nhunit
ftot3 = figure; % figura de Kappas de cada arquitectura
set( ftot3 , 'numbertitle' , 'off' );
set( ftot3 , 'Name' , 'KAPPAS DE LAS CV EN CADA ARQUITECTURA' );
H3 = subplot( 1 , 1 , 1 ); % Handle de los ejes
puntos_linea = 1:1:n_arch;
hold on;
plot( Kappa_arq , ':or');
if size( Kappa_arq_eval , 2 ) == 1 % si es un vector columna, hay que dibujar los puntos uno a uno
    for dd = 1:1:size( Kappa_arq_eval , 1 )
        plot( Kappa_arq_eval( dd ) , 'o' );
    end
else % si es una matriz, se pueden dibujar los puntos a la vez
    plot( Kappa_arq_eval' , 'o' );
end
set( H3 , 'XTick', puntos_linea );
set( H3 , 'XTickLabel' , nhunit );
xlabel('Numero de unidades ocultas');
ylabel( 'Kappas en las CV y Kappa medio' );
axis auto;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% se leen datos de las redes, por conveniencia solo se simulan matrices de
% redes con una unica arquitectura, si aparecen mas de una, se sale de la rutina
n_arch = length( nhunit );
[ n1 n2 n3] = size( mlpnet3 );
if ( n1 ~= n_pool || n2 ~= n_eval || n3 ~=  n_arch )
    error('Existen incongruencias en los datos.');
%     return;
end

while( 1 ) % se sale del bucle no seleccionando ninguna arquitectura
    
    if (n_arch > 1)  % numero de arquitecturas mayor que 1
        fprintf('\n\nSe han encontrado %d arquitecturas de red: ' , n_arch );
        for i = 1:n_arch
            fprintf( '\n%d - Número de neuronas ocultas: %d' , i , nhunit( i ) );
        end
        ind_arch = input('\n\nINTRODUCE EL INDICE DE ARQUITECTURA A SIMULAR: ');
        if isempty( ind_arch )
            break; % salimos del bucle
        end
        if (ind_arch > 0 && ind_arch <= n_arch )
            mlpnet = mlpnet3( : , : , ind_arch ); % extraemos las redes de la arquitectura escojida
        else
            fprintf('\nÍndice de arquitectura erróneo.');
            continue;
        end
    else 
        if ( n_arch == 1 ) % solo una arquitectura
            ind_arch = 1;
            mlpnet = mlpnet3; % extraemos las redes de la unica arquitectura
        else
            break; % numero de arquitecturas es un numero negativo o nulo
        end
    end
    
    if ( n_eval > 1 ) % se salta si N_eval ==1, porque con una sola validacion cruzada este bucle replica resultados por pantalla
        for nm = 1:n_eval
            fprintf('\n\nResultados medio de %i redes en la arq. con %i neuronas ocultas y CV-%i:\n' , n_pool , nhunit(ind_arch) , nm );
            Matriz_Confusion = conf_mat_arq( : , : , nm , ind_arch ); 
            fprintf('\nKappa: %7.6f ;' , Kappa_arq_eval( nm , ind_arch ) );
            fprintf('\nMatriz de confusión:\n' );
            [ num_rows , num_cols ] =  size( Matriz_Confusion );
            for p = 1:num_rows
                fprintf(' %5.0f  ' , Matriz_Confusion( p , : ) );
                fprintf('\n');
            end            
        end
    end
    fprintf('\n\nRESULTADO MEDIO DE LA ARQ. CON %i NEURONAS OCULTAS:\n' , nhunit( ind_arch ) );
    Matriz_Confusion = sum( conf_mat_arq( : , : , : , ind_arch ) , 3 );  % salida por pantalla de la matriz
    Kappa = kappa_index( Matriz_Confusion );  
    Kappa_arq( ind_arch ) = Kappa;
    muestras_nulas = npat - sum( sum( Matriz_Confusion ) );
    fprintf('\nKappa: %7.6f ;' , Kappa );
    fprintf('\nNúmero de muestras nulas: %i ;' , muestras_nulas );
    fprintf('\nMatriz de confusión:\n' );
    [ num_rows , num_cols ] =  size( Matriz_Confusion );
    for p = 1:num_rows
        fprintf(' %5.0f  ' , Matriz_Confusion( p , : ) );
        fprintf('\n');
    end

    Ytot = [];
    out_tot = [];
    ind_tot = [];
    for nm = 1:n_eval
        
        % Generamos dos vectores de indices de muestras:
        %   intest para las muestras de prueba 
        %   intrain  para las muestras de entrenamiento
        intest = ( nm : n_eval : npat );         % indices de test 
        outdatatest = outputdata( : , intest );  % salidas de test
        indatatest = inputdata( : , intest );    % entradas de test
        
        for j = 1:n_pool
            
            %%%%%%%%%   EXTRAEMOS LA RED DE LA COMBINACION: CV Y POOL   %%%%%%%%%%%%%
            if ( n_eval == 1 ) % numero CV es 1
                if ( n_pool == 1 ) % numero pool es 1
                    net = mlpnet{ 1 };
                else % numero pool es mayor que 1
                    net = mlpnet{ j };
                end
            else % numero CV es mayor que 1
                if ( n_pool == 1 ) % numero pool es 1
                    net = mlpnet{ 1, nm };
                else % numero pool es mayor que 1
                    net = mlpnet{ j , nm };
                end
            end     
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%% SIMULANDO LA RED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            Y = sim(net, indatatest); % Obtenemos la respuesta para los patrones de test     
            if (j == 1)
                Ymed = Y;
            else
                Ymed = Ymed + Y;
            end
        end
        
        Ymed = Ymed / n_pool;
        Ytot = [ Ytot  Ymed ];
        out_tot = [ out_tot  outdatatest ];
        ind_tot = [ ind_tot  intest ];
        
    end
    
    [ ind_tot orden ] = sort( ind_tot );  % ordenamos los indices de menor a mayor
    out_tot = out_tot( : , orden );       % ordenamos las salidas deseadas
    Ytot = Ytot( : , orden );             % ordenamos las salidas obtenidas
    
    % Pintamos un histograma de los pesos de todas las redes
    bins = 50; % numero de intervalos en valores de pesos
    [H2 weighs] = weigh_histogram( bins , mlpnet );
    set( H2 , 'name' , ['HISTOGRAMA DE PESOS DE: ' sprintf('%d', n_arch*n_eval*n_pool) ' REDES' ] );
    
    % Pintamos las respuestas en grafico de barras conjuntamente
    base = '1:439 No Reingresa - 440:488 Si Reingresa';
    xticks = 1;
    for n = 1:numclases % contamos las muestras en clases para pintar limites en eje X
        xticks = [ xticks ; sum( numpatclases( 1:n ) ) ]; % Puntos de eje X de las graficas
    end
    titulo = ['CLASIFICACION REINGRESO (' sprintf('Arq. %i neuronas ocultas. CV = %i. Pool = %i redes' , nhunit( ind_arch ) , nm , n_pool ) ')' ];
    H1 = barcompout( Ytot , 'Out_' , base , xticks , titulo );
    
    % Pintamos las respuestas en graficos de barras para cada clase
    % Etiquetas para la representación en eje Y en neuronas
    nombre_salidas  = { 'No Reingresa' , 'Si Reingresa' };
    for k = 1:numout
        nombre_salidas{ ind_salidas( k ) } = [ 'Unit ' num2str( k ) '-' nombre_salidas{ ind_salidas( k ) } ];
    end
    
    % Funcion que pinta graficos de barras para cada clase
    Hbar = bar_out( Ytot' , numpatclases , name_clases , nombre_salidas( ind_salidas ) , umbral );
    
    % Cambiamos el tamaño de las figuras para que se vean mas verticales
    for k = length( Hbar ):-1:1
        set( Hbar( k ) , 'Position' , [ ( 100 * k )   70    600   620 ] );
    end
    
    if ( ind_arch == 1 )
        break; % salimos del bucle porque solo hay una arquitectura
    end
end