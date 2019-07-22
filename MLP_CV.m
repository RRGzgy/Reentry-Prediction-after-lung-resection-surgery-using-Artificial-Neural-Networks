% En este fichero se entrenan varias redes MLP (MultiLayer Perceptrons
% con diferentes arquitecturas, en un problema de reingreso
% Se efectuará un análisis de simulaciones por validación
% cruzada. En cada simulación separamos el conjunto de datos en dos
% subconjuntos: entrenamiento y test. El conjunto de test se determina por
% la matriz de índices "test", mientras el entrenamiento usará el resto de
% las muestras. Repetimos las simulaciones para todas las combinaciones de
% pares de conjuntos diferentes y promediamos los resultados del error,
% para evaluar la generalización obtenida con el modelo neuronal.
clear;
warning off
fprintf('\n\nEn este fichero se entrenan varias redes MLP (MultiLayer Perceptrons)\n');
fprintf('con diferentes arquitecturas, en un problema de reingreso.\n');
fprintf('\nSe efectuará un análisis de simulaciones por validación cruzada.\n');
fprintf('En cada simulación separamos el conjunto de datos en dos subconjuntos: entrenamiento y test.\n');
fprintf('El conjunto de test se determina por la matriz de índices "intest", mientras el entrenamiento \n');
fprintf('usará el resto de las muestras.\n');
fprintf('Repetimos las simulaciones para todas las combinaciones de pares de conjuntos diferentes\n');
fprintf('y promediamos los resultados del error, para evaluar la generalización obtenida con el modelo neuronal.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%      CARGAMOS LOS DATOS DE ENTRADAS Y SALIDAS       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reingreso_balanceado;
%reingreso;
pinta_muestras = 1;       % 1 = pinta el error en muestras
pinta_evaluaciones = 1;   % 1 = pinta el error medio en las evaluaciones
N_OUT = 2; % numero de salidas del problema
error = 0.001; % error maximo permitido


% Indice :  1  2  3  4  5  6  7  8  9  10  11  12 13 14 15 16 17 18 19 20
% 21 22 23 24 25 26
mascara = [ 1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];

indices = find( ( mascara == 1 ) );       % extraemos los indices de variables no enmascaradas
inputdata = data( indices , : );          % Extraemos solamente las variables NO enmascaradas
numinp = sum( mascara );                  % numinp = numero de variables No enmascaradas
names_variables = names( indices );       % nombres de variables NO enmascaradas
clear data;

[ nclases , npat ] = size( clases ); % clases es una matriz dispersa, npat = numero de muestras
out = full( clases );                 % out es una matriz completa
% Calculamos las prob a priori de las clases
prob_priori = numpatclases / npat; 

% Numero de salidas de la red
fprintf('\nÍNDICES DE SALIDAS PARA ENTRENAR LAS REDES' );
ind_salidas = input('\n(por defecto [1 2]): ');
if ( isempty( ind_salidas ) ); 
    ind_salidas = ( 1:1:N_OUT ); % por defecto [ 1 2]
end
outputdata = out( ind_salidas , : );     % seleccionamos las salidas que deseamos que tenga la red
[ numout dummy ] = size( outputdata );   % Numero de clases de salida
ind_out = 1:1:numout;                % vector de indices de salidas
clase = ind_out * outputdata;            % matriz de clasificacion de patrones no-dispersa
prob_priori = prob_priori( ind_salidas );   % seleccionamos las prob a priori de las clases seleccionadas

% Valor del umbral con el que comparar las salidas de la red
umbral = input('\n\nVALOR DE UMBRAL DE SALIDA DE LA RED (por defecto 0.1): ');
if ( isempty( umbral ) || ( umbral < 0 ) ); 
    umbral = 0.1; % umbral para seleccionar una salida de red como no nula
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%               INTRODUCIENDO DATOS DE ENTRENAMIENTO                       %%%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unidades Ocultas en las diferentes arquitecturas
nhunit = input('\n\nNÚMERO DE UNIDADES OCULTAS EN LAS ARQUITECTURAS (por defecto [12]): ');
if ( isempty( nhunit ) ); nhunit = 12; end

n_arch = length( nhunit ); % numero de arquitecturas
n_pool = input('\n\nNÚMERO DE REPETICIONES DE CADA RED (por defecto 20): ');
if ( isempty( n_pool ) || n_pool <= 0 ); n_pool = 20; end

% Preguntamos el método de entrenamiento
funtrain = pidefuntrain;
if isempty( funtrain ); 
    funtrain = 'trainlm'; % por defecto L-M
end 

% Número máximo de ciclos de entrenamiento
% (disminuirlo para los metodos cuasi-Newton y de grad.conjugados)
numciclos = input('\n\nNÚMERO MÁXIMO DE CICLOS DE ENTRENAMIENTO (L-M=30): ');
if isempty( numciclos ); numciclos = 30; end 

% Introducimos el numero de validaciones cruzadas
n_eval = input('\n\nINTRODUCE EL NUMERO DE VALIDACIONES CRUZADAS (por defecto 8): ');
if ( isempty( n_eval ) || n_eval <= 0 || n_eval > npat ); n_eval = 8; end 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                AQUI SE ENTRENAN Y EVALUAN LAS REDES                      %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off MATLAB:nearlySingularMatrix;
Toterr = [];
conf_mat_arq = []; % matriz 4D para guardar las matrices de confusion promedio de cada arquitectura y evaluacion
HBar = zeros( n_arch ); % handles de figuras
Kappa_arq_eval = zeros( n_arch , n_eval ); % Kappas de cada validación en las arquitecturas
Toterr_arq = zeros( 1 , n_arch ); % promedio de errores de todas las evaluaciones 
Totstd_arq = zeros( 1 , n_arch ); % desviacion estandar de todas las evaluaciones
min_error = zeros( 1 , n_arch );  % minimo error de las redes del pool en esta arquitectura
max_error = zeros( 1 , n_arch );  % maximo error de las redes del pool en esta evaluacion

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                    BUCLE DE ARQUITECTURAS DE RED                         %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:n_arch  % bucle en las diferentes arquitecturas
    
    if pinta_muestras
        % pintamos las graficas horizontales del error promedio en cada muestra
        HBar(i) = figure; % figura de errores por patrones
        set( HBar(i) , 'numbertitle' , 'off' );
        set( HBar(i) , 'name' , sprintf( 'ARQUITECTURA CON %d NEURONAS OCULTAS: ERRORES CUADRÁTICOS DE MUESTRAS DE PRUEBA POR CADA EVALUACION Y RED DEL POOL' , nhunit(i) ) );
    end
    
    conf_mat_eval = []; % matriz 3D para guardar las matrices de confusion promedio de cada evaluacion
    Kappa_eval = zeros( 1, n_eval ); % vector para guardar kappas de cada validacion
    Toterr_pool = zeros( 1 , n_eval ); % promedio del error de todas las muestras en esta evaluacion
    Totstd_pool = zeros( 1 , n_eval ); % desviacion estandar de error de los errores de esta evaluacion
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%              INICIO DE BUCLE DE EVALUACIONES CV                   %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Generamos n_eval conjuntos de indices de muestras para test,
    % consistentes en incluir las muestras de test cada n_eval muestras
    % del conjunto, dejando las demas para el entrenamiento
    % Si se dispusieran de muchas muestras, seria conveniente hacer una
    % ordenacion aleatoria de las muestras, previa a esta seleccion.
    % En este ejemplo, lo hacemos asi, ya que se disponen de muy pocas muestras
    % de cada clase, y no conviene que al hacer la aleatorizacion previa, 
    % todas las muestras de una misma clase pasen a test o entrenamiento.
    for nm = 1:n_eval % bucle en las diferentes evaluaciones de CV
        
        % Generamos dos vectores de indices de muestras:
        % Los conjuntos de evaluación, dado que el número de patrones es bajo, serán adjudicados
        % por el usuario de la siguiente forma: se introduce en datos una matriz de índices "intest"
        % donde cada fila será un conjunto de muestras de test y el correspondiente conjunto de train serán
        % el resto de patrones. 
        %   intest para las muestras de prueba 
        %   intrain  para las muestras de entrenamiento
        fprintf('\nSe selecciona el grupo de muestras de la Validacion Cruzada: %d\n' , nm ); 
        
        intest = ( nm : n_eval : npat );         % indices de test 
        outdatatest = outputdata( : , intest );  % salidas de test
        indatatest = inputdata( : , intest );    % entradas de test
        
        intrain = [];                            % indices de train 
        for m = 1:n_eval
            if m ~= nm; 
                intrain = cat( 2 , intrain , ( m : n_eval : npat ) ); 
            end
        end
        % si solo hay una validacion cruzada, el conjunto de test y de train son el mismo
        if (n_eval == 1) 
            intrain = intest;
        end
        outdatatrain = outputdata( : , intrain ); % salidas de train
        indatatrain = inputdata( : , intrain );   % entradas de train
        
        %Crear funcion para balancear resultados
       
        
        datam = [ min( indatatrain, [], 2 ) , max( indatatrain, [], 2 ) ]; % valores minimos y maximos en las entradas
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%       BUCLE DE POOL DE REDES PARA CADA ARQ Y EVALUACION CV        %%%                       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:n_pool  
            
            % Entrenamos la j-esima red a evaluar del pool de redes
            % con la arquitectura i-esima 
            % en el conjunto de CV nm-esimo            
            if ( nhunit( i ) )                
                net = newff( datam , [ nhunit( i ) numout ] , { 'tansig' , 'logsig' } , funtrain );
            else
                net = newff( datam , numout , { 'logsig' } , funtrain );
            end
            
            % Número de ciclos de entrenamiento 
            net.trainParam.epochs = numciclos;
            
            % Error cuadrado máximo promedio
            net.trainParam.goal = error; % error maximo permitido
            
            % Entrenamos a la Red
            fprintf('\nARQUITECTURA CON %d NEURONAS OCULTAS: VALIDACION CRUZADA %d, ENTRENANDO LA RED %d\n' , nhunit( i ) , nm , j );
            net = train( net , indatatrain , outdatatrain );
            
            % Obtenemos la respuesta para los patrones de test
            Y = sim( net , indatatest ); 

            % Calculamos el vector de errores en cada patron frente a la respuesta deseada
            err = sum( ( ( Y - outdatatest ).^ 2 ) , 1 ); % sumamos el error de todas las salidas de la red
            if ( j == 1 )
                mlpnet = { net }; % guardamos los datos de la red en un vector de redes
                errnet = err; % guardamos el error medio de salida de cada patron en esta evaluacion
                Ymed = Y; % guardamos las salidas
            else
                mlpnet = cat( 1 , mlpnet , {net} );  % acumulamos los datos de la red en un vector de redes
                errnet = cat( 1 , errnet , err );  % guardamos los vectores de error de evaluaciones por filas
                Ymed = Ymed + Y; % acumulamos las salidas de las evaluaciones
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%     MEDIDAS DE LOS ERRORES EN TODAS LAS MUESTRAS DE EVALUACION    %%%                       
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if pinta_muestras           
                
                % pintamos las graficas horizontales del error promedio en cada muestra
                figure( HBar( i ) );
                Hj = subplot( n_eval , n_pool , ( nm - 1 ) * n_pool + j );
                bar( intest , err );
                axis tight;
                set( Hj , 'XTick' , intest );
                set( Hj , 'XTickLabel' , intest ); % etiquetas numericas en la base de la grafica
                grid;
          
                if ( j == 1 )
                    ylabel( sprintf( 'CV-%d' , nm ) );
                end
                if ( j == ceil( n_pool/2 ) && nm == n_eval )
                    xlabel( 'REDES DEL POOL - INDICES DE MUESTRAS DE TEST' );
                end                
                
            end

        end  % fin de bucle de j para n_pool redes con la misma arquitectura y la misma CV
        
        % Promediamos las salidas acumuladas del pool de redes
        Ymed = Ymed / n_pool; 
        
        % Obtenemos una matriz 'clasificacion' de 1 y 0 para las salidas seleccionadas como
        % maximas que esten por encima del umbral, si no seran todas cero 
        clasificacion = clasificacion_MLP( Ymed , prob_priori', umbral );
        
        % Calculamos la matriz de confusion. Si una columna de
        % clasificacion es todo ceros, aparece una clase de rechazo
        [ conf_mat , Kappa ] = confussion_matrix( ind_out * clasificacion , clase( intest ) );

        errnetmedio_pool = mean( errnet , 2 ); % error medio en cada patron en las n_pool simulaciones
        Toterr_pool(nm) = mean( errnetmedio_pool ); % promediamos los errores de todas las muestras en esta evaluacion
        Totstd_pool(nm) = std( errnetmedio_pool ); % desviacion estandar de error de los errores medios en cada muestra de esta evaluacion

        % Guardamos el vector de redes en una matriz de redes
        if ( nm == 1 )
            mlpnet2 = mlpnet;
            conf_mat_eval = conf_mat;
        else
            mlpnet2 = cat( 2 , mlpnet2 , mlpnet );   
            conf_mat_eval = cat( 3 , conf_mat_eval , conf_mat );
        end
        Kappa_eval( nm ) = Kappa; 
    
    end  % fin del bucle for de nm validaciones cruzadas
    
    %%%%%  PINTANDO Error medio en cada pool  %%%%%  
    if pinta_evaluaciones
        ftot = figure; % figura de Error de cada arquitectura en sus CV
        set( ftot , 'numbertitle' , 'off' );
        set( ftot , 'name' , sprintf('NHU-%i: RESULTADOS MEDIOS DE CV EN LOS DIFERENTES CONJUNTOS TEST ', nhunit( i ) ) );
        puntos_linea = 1:1:n_eval;
        bar( Toterr_pool , 'r')
        hold on;
        errorbar( puntos_linea , Toterr_pool , Totstd_pool  , ':ob' )
        legend(  'Medias' , 'Medias +/- DesvStd' );
        xlabel('Indice de Validaciones Cruzadas');
        ylabel( 'Estadisticas del error cuadratico medio' );
        axis auto;
        hold off;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    % Guardamos la matriz de redes en una matriz3D de redes
    if ( i == 1 )
        mlpnet3 = mlpnet2;
        conf_mat_arq = conf_mat_eval;
    else
        mlpnet3 = cat( 3, mlpnet3, mlpnet2 );
        conf_mat_arq = cat( 4 , conf_mat_arq , conf_mat_eval );
    end
    
    Kappa_arq_eval( i , :) = Kappa_eval;        
    Toterr_arq( i ) = mean( Toterr_pool ); % promediamos los errores de todas las evaluaciones 
    Totstd_arq( i ) = std( Toterr_pool ); % desviacion estandar de todas las evaluaciones
    min_error( i ) = min( Toterr_pool ); % guardamos el minimo error de las redes del pool en esta evaluacion
    max_error( i ) = max( Toterr_pool ); % guardamos el maximo error de las redes del pool en esta evaluacion
    
end % fin del bucle i para las diferentes arquitecturas 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                      GUARDAMOS LAS REDES RESULTANTES                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% y un segundo con nombre descriptivo: numero de arquitecturas, funcion de entrenamiento,
% numciclos, numero CV y pool  
nombre_fichero = [ 'MLP_CV_reingreso_n_arq_' num2str( n_arch ) '_numout_' num2str( numout ) '_funtrain_' funtrain '_ciclos_' num2str( numciclos ) '_evalCV_' num2str( n_eval ) '_pool_' num2str( n_pool ) '.mat ' ]
eval( [ 'save ' nombre_fichero  ] );
save MLP_CV_reingreso