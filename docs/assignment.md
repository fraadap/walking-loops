Realizza un progetto Python di Reinforcement Learning focalizzato sulla progettazione di un environment Gym compatibile con PPO di Stable-Baselines3.

Obiettivo generale

Costruire un environment che generi percorsi pedonali a loop su una mappa reale ottenuta da OpenStreetMap.

Il comportamento desiderato è:

il percorso parte da un punto fisso
visita eventualmente alcuni waypoint intermedi
ritorna automaticamente al punto di partenza
la lunghezza totale del loop deve avvicinarsi il più possibile a una durata target espressa in minuti

In questa prima versione il focus è:

semplicità architetturale
stabilità dell’environment
osservabilità del comportamento
visualizzazione grafica immediata del loop generato

Non è necessario implementare subito personalizzazione utente, POI avanzati o reward complessi.

Stack tecnico

Usare:

Python
Gymnasium (o Gym compatibile)
Stable-Baselines3
OSMnx
NetworkX
Matplotlib
NumPy
Area geografica iniziale

Usare inizialmente un’area geografica piccola, in modo da ridurre complessità computazionale.

Preferibilmente un piccolo quartiere o una piccola località.

L’area deve essere facilmente sostituibile in futuro con aree più grandi.

Grafo della mappa

Scaricare il grafo pedonale da OpenStreetMap tramite OSMnx.

Requisiti:

grafo walkable
nodi = intersezioni / punti navigabili
archi = percorsi pedonali
peso principale = lunghezza in metri

Il grafo deve essere memorizzato come stato persistente dell’environment.

Punto di partenza

Per questa prima versione:

usare un nodo fisso
il nodo iniziale deve essere scelto una sola volta all’inizio

In una futura estensione:

il punto iniziale potrà essere la posizione attuale dell’utente
Rappresentazione interna del loop

Il loop deve essere rappresentato come:

nodo di partenza
lista ordinata di waypoint intermedi

Non memorizzare direttamente il percorso completo.

Il percorso completo deve essere ricostruito dinamicamente usando shortest path.

Costruzione del loop

Ogni volta che il loop viene valutato:

partire dal nodo iniziale
attraversare i waypoint nell’ordine corrente
calcolare shortest path tra ogni coppia consecutiva
chiudere automaticamente il loop tornando al nodo iniziale

Importante:

il tempo di ritorno allo start deve essere sempre incluso nella lunghezza totale del percorso.

Se il loop è vuoto:

il percorso è semplicemente start → start
lunghezza zero
Conversione distanza → durata

Assumere velocità di camminata costante:

5 km/h

Convertire distanza in metri in minuti.

Formula:

durata_minuti = distanza_metri / 1000 / 5 * 60

Action space

Usare action space discreto.

Azioni:

0 = aggiungi waypoint
1 = rimuovi waypoint
2 = fermati
Regole
ADD

Aggiunge un nuovo waypoint candidato.

REMOVE

Rimuove l’ultimo waypoint della lista.

Se non ci sono waypoint:

non genera errore
stato invariato
STOP

Termina l’episodio.

Limitazione dello spazio delle azioni

Non permettere all’agente di scegliere direttamente qualunque nodo del grafo.

Implementare invece un insieme limitato di waypoint candidati.

Strategia richiesta

All’inizializzazione:

campionare un piccolo sottoinsieme di nodi del grafo
usare ad esempio 10–20 nodi candidati

Vincoli desiderabili:

nodi non troppo vicini tra loro
nodi raggiungibili dal nodo iniziale
nodi distribuiti spazialmente

Per la prima versione è sufficiente un semplice campionamento casuale controllato.

Quando viene eseguita azione ADD:

selezionare un waypoint candidato non ancora presente nel loop
la selezione può essere casuale

Questo mantiene lo spazio di controllo gestibile da PPO.

Observation space

L’osservazione deve essere un vettore numerico.

Deve contenere almeno queste feature:

durata target
durata attuale del loop
errore assoluto rispetto al target
numero di waypoint
percentuale di waypoint ripetuti (inizialmente può essere zero)
numero di nodi del loop
numero massimo di step residui
Normalizzazione

Le feature devono essere normalizzate in range ragionevoli.

Esempio:

target tra 5 e 50 minuti
current_length diviso 50
num_waypoints diviso max_waypoints
Target di durata

All’inizio di ogni episodio:

campionare casualmente una durata target tra 5 e 50 minuti.

Questo obbliga l’agente a generalizzare.

Reward function

Per questa prima versione usare reward semplice.

Obiettivo unico:

minimizzare errore rispetto alla durata target.

Formula consigliata:

reward = - abs(current_duration - target_duration)

Nota importante

Non aggiungere ancora:

bonus POI
bonus estetici
penalità topologiche
reward umani

La priorità è stabilizzare l’ambiente.

Termination conditions

Un episodio termina se accade una delle seguenti condizioni:

STOP

L’agente sceglie fermarsi.

Errore troppo grande

Se la durata corrente supera troppo il target.

Per esempio:

current_duration > target_duration * 1.8
Numero massimo di step

Imporre limite massimo.

Per esempio:

10 o 15 step
Metriche di training

Durante training registrare:

reward medio per episodio
errore medio rispetto al target
durata finale del loop
numero di waypoint usati
Visualizzazione grafica

Questa parte è importante.

Alla fine di ogni episodio (oppure ogni N episodi), generare una visualizzazione della mappa.

La visualizzazione deve mostrare:

grafo stradale di base
nodo iniziale
waypoint selezionati
loop finale completo
Requisiti visuali

Usare Matplotlib + OSMnx.

La figura deve permettere di capire immediatamente:

se il loop è sensato
se il loop è troppo corto
se il loop è troppo lungo
se l’agente sta migliorando

Stampare anche:

target duration
achieved duration
reward finale
PPO

Usare Stable-Baselines3 PPO.

Configurazione iniziale semplice.

Non serve tuning aggressivo.

L’obiettivo è ottenere:

ambiente funzionante
primi episodi
prima curva di apprendimento

Allenare inizialmente almeno 50 episodi.

Struttura del codice desiderata

Organizzare il progetto in modo modulare.

walking_env.py

Contiene:

classe environment
reset()
step()
observation
reward
rendering
train.py

Contiene:

creazione environment
training PPO
logging metriche
visualizzazione risultati
utils.py

Funzioni di supporto:

shortest path
conversione metri → minuti
sampling waypoint
plotting
Obiettivi di questa prima milestone

Alla fine della prima implementazione il progetto deve permettere di:

costruire environment funzionante
lanciare PPO
eseguire episodi
generare loop reali su mappa
visualizzare graficamente il comportamento
Vincoli progettuali importanti

Preferire:

semplicità
leggibilità
modularità
facilità di debugging

Evitare premature ottimizzazioni.

La priorità non è performance massima, ma correttezza concettuale.