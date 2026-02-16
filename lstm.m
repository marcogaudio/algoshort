%% Dataset di serie temporali allo scopo di classificare attività umane
%% tramite una rete neurale basata su LSTM e TCN.
% Questo script carica dati accelerometrici (x, y, z) raccolti da sensori
% indossati da 24 soggetti durante diverse attività (camminare, correre,
% salire/scendere scale, ecc.), li segmenta in finestre temporali,
% addestra una rete LSTM per classificare l'attività, e ne valuta l'accuratezza.


%%DATASET DESCRIZIONE

% Lettura di un singolo file CSV di esempio per visualizzare i dati grezzi.
% Il file contiene le colonne: [timestamp, acc_x, acc_y, acc_z].
d=readtable("B_Accelerometer_data/dws_11/sub_1.csv");

% Crea una nuova finestra grafica (vuota) dove verrà disegnato il plot.
figure();

% Converte la tabella MATLAB in una matrice numerica standard,
% necessaria per poter usare operazioni matriciali e indicizzazione diretta.
dt=table2array(d);

% Disegna le 3 colonne (acc_x, acc_y, acc_z) su un unico grafico.
% La colonna 1 (timestamp) viene ignorata perché non serve per la visualizzazione.
% Ogni colore rappresenta un asse dell'accelerometro.
plot(dt(:,2:4));

% Imposta il percorso radice della cartella che contiene tutte le sottocartelle
% dei dati accelerometrici. Ogni sottocartella corrisponde a un'attività diversa
% (es. "dws" = downstairs, "ups" = upstairs, "wlk" = walking, ecc.).
root='./B_Accelerometer_data';

% Definisce l'array dei soggetti: 24 persone numerate da 1 a 24.
% Ogni soggetto ha un file CSV separato per ogni attività.
subjects=1:24;

% Frequenza di campionamento del sensore: 50 Hz (50 campioni al secondo).
smpf=50;

% Lunghezza di ogni sequenza temporale: 5 secondi * 50 Hz = 250 campioni.
% Ogni finestra di 250 campioni diventerà un singolo esempio di input per la rete.
seqlen=5*smpf;

% Elenca tutte le sottocartelle dentro "B_Accelerometer_data/".
% Ogni cartella rappresenta un tipo di attività (es. "dws_11", "ups_12", "wlk_7").
% La funzione dir() restituisce una struct con campi: name, folder, date, bytes, isdir.
folders=dir("B_Accelerometer_data/*");


% Inizializza due cell array vuoti:
% - xdata conterrà le sequenze di input (matrici 250x3 di accelerazioni)
% - ydata conterrà le etichette corrispondenti (stringhe di 3 caratteri come "dws", "ups", "wlk")
xdata=[];
ydata=[];

% Ciclo esterno: scorre ogni cartella (ogni attività) trovata nella directory.
% numel(folders) restituisce il numero totale di elementi nella struct folders.
for f=1:numel(folders)

    % Controlla che il nome della cartella NON inizi con '.' (punto).
    % Questo filtra via le cartelle speciali "." (corrente) e ".." (genitore)
    % che vengono sempre restituite da dir() su sistemi Unix/Mac.
    if folders(f).name(1)~='.'

        % Ciclo intermedio: scorre ogni soggetto da 1 a 24.
        for s=1:numel(subjects)

            % Costruisce il percorso completo del file CSV.
            % Esempio risultante: "./B_Accelerometer_data/dws_11/sub_1.csv"
            % filesep inserisce il separatore di cartelle del sistema operativo
            % ('/' su Mac/Linux, '\' su Windows).
            % num2str() converte il numero del soggetto in stringa.
            fname=[root filesep folders(f).name filesep 'sub_' num2str(subjects(s)) '.csv'];

            % Legge il file CSV come tabella MATLAB.
            % Ogni riga contiene: [timestamp, acc_x, acc_y, acc_z].
            data=readtable(fname);

            % Calcola quante sequenze complete di lunghezza seqlen (250 campioni)
            % possono essere estratte dal file.
            % fix() arrotonda per difetto: se il file ha 1100 righe,
            % numseq = fix(1100/250) = 4 sequenze (i restanti 100 campioni vengono scartati).
            numseq=fix(size(data,1)/seqlen);

            % Ciclo interno: estrae ogni sequenza dal file.
            for i=1:numseq

                % Estrae la i-esima finestra temporale di 250 campioni.
                % Prende solo le colonne 2:4 (acc_x, acc_y, acc_z), escludendo il timestamp.
                % L'indice di riga va da (1+(i-1)*250) a (i*250):
                %   - i=1: righe 1-250
                %   - i=2: righe 251-500
                %   - i=3: righe 501-750, ecc.
                % table2array() converte la sotto-tabella in matrice numerica 250x3.
                % {end+1,1} aggiunge l'elemento in coda al cell array xdata.
                xdata{end+1,1}=table2array(data(1+(i-1)*seqlen:i*seqlen,2:4));

                % Salva l'etichetta dell'attività: i primi 3 caratteri del nome della cartella.
                % Esempi: "dws" (downstairs), "ups" (upstairs), "wlk" (walking),
                %         "jog" (jogging), "sit" (sitting), "std" (standing).
                % Questo associa ogni sequenza alla sua classe di appartenenza.
                ydata{end+1,1}=folders(f).name(1:3);

            end
        end
    end
end


%% Rete
% ===== PREPARAZIONE DEI DATI PER L'ADDESTRAMENTO =====

% Converte il cell array di stringhe ydata in un array categoriale.
% Le variabili categoriali sono il formato richiesto da MATLAB per le etichette
% di classificazione. Internamente assegna un indice numerico ad ogni classe.
ydata=categorical(ydata);

% Divide casualmente gli indici dei dati in 3 sottoinsiemi:
%   - trainInd: 70% dei dati per l'addestramento (la rete impara i pesi)
%   - valInd:   10% dei dati per la validazione (monitoraggio dell'overfitting durante il training)
%   - testInd:  20% dei dati per il test finale (valutazione delle prestazioni)
% La divisione casuale garantisce che ogni sottoinsieme sia rappresentativo.
[trainInd,valInd,testInd]=dividerand(numel(ydata),0.7,0.1,0.2);

% Estrae le sequenze di input (accelerazioni) per ogni sottoinsieme
% usando gli indici generati dalla divisione casuale.
xtrain=xdata(trainInd);   % Sequenze per addestramento
xtest=xdata(testInd);     % Sequenze per test finale
xval=xdata(valInd);       % Sequenze per validazione

% Estrae le etichette corrispondenti per ogni sottoinsieme.
% Ogni etichetta è associata alla stessa sequenza tramite lo stesso indice.
ytrain=ydata(trainInd);   % Etichette per addestramento
yval=ydata(valInd);       % Etichette per validazione
ytest=ydata(testInd);     % Etichette per test finale


% Stampa un riepilogo statistico dell'intero dataset:
% mostra quanti campioni ci sono per ogni classe (dws, ups, wlk, ecc.).
% Utile per verificare se il dataset è bilanciato o sbilanciato.
summary(ydata);

% Stampa lo stesso riepilogo ma solo per il set di test.
% Serve a verificare che anche il test set contenga tutte le classi
% in proporzioni ragionevoli.
summary(ytest);

%% Costruzione della rete
% ===== DEFINIZIONE DELL'ARCHITETTURA DELLA RETE NEURALE LSTM =====

% Estrae la lista di tutte le classi presenti nel dataset.
% Restituisce un cell array di stringhe: {"dws", "jog", "sit", "std", "ups", "wlk"}.
classes=categories(ydata);

% Trova i nomi unici delle classi nel set di training.
% Simile a categories() ma lavora direttamente sul sottoinsieme di addestramento.
classNames=unique(ytrain);

% Numero di canali (features) in input: 3, corrispondenti agli assi
% x, y e z dell'accelerometro. Ogni timestep ha 3 valori.
numChans=3;

% Numero totale di classi da predire (es. 6 attività diverse).
% Determina la dimensione dello strato di output della rete.
numClasses=numel(classNames);

% Dimensione del mini-batch: 128 sequenze vengono processate insieme
% prima di ogni aggiornamento dei pesi (backpropagation).
% Valori tipici: 32, 64, 128, 256.
% Batch più grandi = training più veloce ma più memoria richiesta.
miniBatchSize=128;

% Numero massimo di epoche: 20 passaggi completi attraverso tutto il training set.
% Un'epoca = la rete ha visto ogni singolo esempio almeno una volta.
maxEpochs=20;

% Numero di unità nascoste (neuroni) in ogni layer LSTM: 128.
% Questo definisce la capacità della rete di memorizzare pattern temporali.
% Più neuroni = maggiore capacità ma rischio di overfitting e training più lento.
numHiddenUnits=128;


% Definizione dell'architettura della rete come array di layer sequenziali.
% I dati passano dal primo all'ultimo strato in ordine.
layers=[

    % STRATO 1 - Input: accetta sequenze temporali con 3 features per timestep.
    % Dimensione input: [3 x T] dove T è la lunghezza della sequenza (250).
    sequenceInputLayer(3), ...

    % STRATO 2 - Primo LSTM: 128 neuroni nascosti.
    % "OutputMode"="sequence" significa che produce un output per OGNI timestep,
    % non solo per l'ultimo. Questo è necessario perché il prossimo LSTM
    % ha bisogno dell'intera sequenza in input.
    % L'LSTM mantiene una memoria interna (cell state) che gli permette di
    % catturare dipendenze temporali a lungo termine nei dati.
    lstmLayer(numHiddenUnits,"OutputMode","sequence"), ...

    % STRATO 3 - Dropout al 40%: durante l'addestramento, disattiva casualmente
    % il 40% dei neuroni ad ogni iterazione. Questo è una tecnica di regolarizzazione
    % che previene l'overfitting costringendo la rete a non dipendere
    % da singoli neuroni specifici. Durante il test, tutti i neuroni sono attivi.
    dropoutLayer(0.4), ...

    % STRATO 4 - Secondo LSTM: 128 neuroni nascosti.
    % "OutputMode"="last" significa che restituisce SOLO l'output dell'ultimo timestep.
    % Questo comprime l'intera sequenza temporale in un singolo vettore di 128 valori,
    % che rappresenta un "riassunto" dell'intera finestra di 5 secondi.
    lstmLayer(numHiddenUnits,"OutputMode","last"), ...

    % STRATO 5 - Fully Connected: mappa il vettore di 128 valori in numClasses valori.
    % Ogni neurone di output corrisponde a una classe (attività).
    % I pesi di questo strato vengono appresi durante l'addestramento.
    fullyConnectedLayer(numClasses), ...

    % STRATO 6 - Softmax: converte i valori grezzi (logits) in probabilità.
    % L'output è un vettore di numClasses valori tra 0 e 1 che sommano a 1.
    % La classe con probabilità più alta è la predizione della rete.
    % Esempio output: [0.01, 0.02, 0.05, 0.01, 0.01, 0.90] -> classe 6 (wlk).
    softmaxLayer ];


% ===== CONFIGURAZIONE DELLE OPZIONI DI ADDESTRAMENTO =====
options=trainingOptions("adam", ...
    ... % "adam" = Adaptive Moment Estimation, ottimizzatore che combina
    ... % i vantaggi di RMSProp e SGD con momentum. Adatta automaticamente
    ... % il learning rate per ogni parametro. È lo standard per reti neurali.

    MaxEpochs=maxEpochs, ...
    ... % Numero massimo di epoche: 20. L'addestramento si ferma dopo 20 passaggi
    ... % completi sul training set (o prima se si usa early stopping).

    MiniBatchSize=miniBatchSize, ...
    ... % Dimensione del mini-batch: 128 sequenze per aggiornamento dei pesi.
    ... % Ad ogni iterazione, la rete processa 128 sequenze, calcola il gradiente
    ... % medio della loss, e aggiorna i pesi tramite backpropagation.

    ValidationData={xval,yval}, ...
    ... % Specifica i dati di validazione. Durante il training, la rete viene
    ... % periodicamente valutata su questi dati per monitorare l'overfitting.
    ... % Se la loss di validazione inizia a salire mentre quella di training scende,
    ... % la rete sta memorizzando i dati invece di generalizzare.

    ValidationFrequency=50, ...
    ... % La validazione viene eseguita ogni 50 iterazioni (non epoche).
    ... % Un'iterazione = un mini-batch processato. Questo determina la frequenza
    ... % con cui i punti di validazione appaiono nel grafico di training.

    InitialLearnRate=0.01, ...
    ... % Tasso di apprendimento iniziale: 0.01. Controlla quanto i pesi vengono
    ... % modificati ad ogni aggiornamento. Troppo alto = training instabile,
    ... % troppo basso = training lentissimo. 0.01 è un buon punto di partenza.

    Plots="training-progress", ...
    ... % Mostra una finestra grafica in tempo reale durante l'addestramento
    ... % con le curve di loss e accuracy per training e validazione.
    ... % Permette di monitorare visivamente il progresso e l'overfitting.

    LearnRateSchedule="piecewise", ...
    ... % Strategia di riduzione del learning rate: "piecewise" significa che
    ... % il learning rate viene moltiplicato per un fattore di riduzione
    ... % ad intervalli regolari. Questo aiuta la rete a convergere:
    ... % all'inizio passi grandi per esplorare, poi passi piccoli per affinare.

    LearnRateDropFactor=0.9);
    ... % Fattore di riduzione del learning rate: 0.9.
    ... % Ad ogni intervallo, il nuovo LR = vecchio LR * 0.9.
    ... % Esempio: 0.01 -> 0.009 -> 0.0081 -> 0.00729 ...
    ... % La riduzione graduale permette un fine-tuning progressivo.


% Analizza e visualizza l'architettura della rete: mostra un diagramma
% con tutti gli strati, le dimensioni dei tensori che passano tra uno strato
% e l'altro, e il numero totale di parametri apprendibili.
analyzeNetwork(layers);

% ===== ADDESTRAMENTO DELLA RETE =====
% Addestra la rete usando:
%   - xtrain: sequenze di input per l'addestramento (cell array di matrici 250x3)
%   - ytrain: etichette corrispondenti (array categoriale)
%   - layers: architettura della rete definita sopra
%   - 'crossentropy': funzione di loss per classificazione multi-classe.
%     Misura quanto le probabilità predette dalla rete sono distanti dalle
%     etichette vere. Più è bassa, migliore è la predizione.
%   - options: tutte le opzioni di training configurate sopra.
% Restituisce 'net': la rete addestrata con i pesi ottimizzati.
net=trainnet(xtrain,ytrain,layers,'crossentropy',options);

%% Valutazione della performance
% ===== TEST DELLA RETE ADDESTRATA =====

% Esegue la predizione su tutte le sequenze del test set.
% minibatchpredict processa i dati in mini-batch (efficiente per grandi dataset).
% YTest è una matrice [N x numClasses] dove ogni riga contiene le probabilità
% per ogni classe. N = numero di sequenze nel test set.
YTest=minibatchpredict(net,xtest);

% Decodifica le probabilità in etichette categoriali.
% onehotdecode prende la matrice di probabilità e per ogni riga seleziona
% la classe con il valore più alto (argmax).
% Parametri:
%   - YTest: matrice di probabilità [N x numClasses]
%   - string(classNames): nomi delle classi come stringhe
%   - 2: dimensione lungo cui cercare il massimo (colonne = classi)
%   - 'categorical': tipo di output desiderato
% Risultato: Ytest è un vettore categoriale di N predizioni.
Ytest = onehotdecode(YTest, string(classNames),2,'categorical');

% Calcola l'accuratezza: percentuale di predizioni corrette.
% Ytest==ytest produce un vettore logico (1 dove corretto, 0 dove sbagliato).
% sum() conta i casi corretti.
% numel(ytest) è il numero totale di campioni nel test set.
% Esempio: se 180 su 200 sono corretti, acc = 180/200 = 0.90 (90%).
acc=sum(Ytest==ytest)/numel(ytest);

% Stampa l'accuratezza formattata con 2 decimali.
% %2.2f stampa un numero con almeno 2 cifre e 2 decimali.
% \n va a capo. Esempio output: "Accuracy on test set was 0.92"
fprintf("Accuracy on test set was %2.2f\n",acc);
