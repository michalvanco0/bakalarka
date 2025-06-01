# Punctuation Network Analyzer
Punctuation Network Analyzer je desktopová aplikácia slúžiaca na analýzu textov pomocou sieťovej reprezentácie interpunkčných a slovných štruktúr. Umožňuje vizualizáciu, štatistickú analýzu a export výstupov do viacerých formátov.

## Spustenie aplikácie
Skontrolujte, že máte nainštalovaný Python 3.10+ a potrebné knižnice:
p i p i n s t a l l −r r e q u i r e m e n t s . t x t
Medzi hlavné závislosti patria: PyQt6, networkx, reportlab.
Spustite aplikáciu príkazom:
python main . py
Otvorí sa hlavné okno aplikácie.

## Výber vstupného súboru
Kliknite na tlačidlo SEARCH FILES.
Vyberte .txt súbor s textom, ktorý chcete analyzovať.
Po výbere sa jeho cesta zobrazí v hornej časti aplikácie.

## Nastavenie interpunkcie
Vyberte, ktoré interpunkčné znaky sa majú analyzovať:
ALL_PUNCTUATION – predvolené, analyzuje všetku interpunkciu.
BASIC – analyzuje základné znaky (. , ! ?).
Custom – umožňuje zadať vlastný zoznam znakov (napr. .,!?-).
Pri výbere „Custom“ sa aktivuje vstupné pole pre manuálny zápis znakov.

## Dodatočné nastavenia vizualizácie a analýzy
Pomocou zaškrtávacích polí môžete aktivovať rôzne výstupy:
 - Words in the network - Zobrazí menovky uzlov v sieti
 - Word network - Vykreslí slovnú sieť
 - Degree distribution - Zobrazí rozdelenie stupňov uzla
 - Binned degree distribution - Zobrazí log binované rozdelenie stupňov uzla
 - Models - Zobrazí porovnanie s modelmi (Weibullovo, Geometrické, Poissonovo)
 - Fit convergence - Vykreslí graf smernice v závislosti od počtu analyzovaných slov
 - Weibull Distribution - Zobrazí Weibullovu distribúciu pre danú sieť
 - Distribution comparison - Porovná rôzne distribúcie (Weibullova, Geometrická,
Poissonova)

#### Filter podľa stupňa
Pomocou posuvníka Min. degree nastavíte minimálny stupeň uzla. Pri vizualizácii
slovnej siete sa zobrazia len tie uzly, ktoré majú minimálne takýto stupeň.

## Spustenie analýzy
Po výbere súboru a nastavení možností kliknite na tlačidlo Analyze text.
Aplikácia vykoná:
 - Tokenizáciu textu (s interpunkciou a bez nej)
 - Vytvorenie dvoch sietí: G1 – so započítanou interpunkciou; G2 – bez interpunkcie
 - Výpočet štatistických a lingvistických metrík
 - Vizualizácie podľa zvolených nastavení
 - Výstup zobrazí v pravej časti oknaMANUÁL K APLIKÁCII PUNCTUATION NETWORK ANALYZER 38

## Export výsledkov
Po úspešnej analýze môžete výsledky exportovať:
 - Export G1 into CSV: Exportuje graf s interpunkciou do .csv
 - Export G2 into CSV: Exportuje graf bez interpunkcie do .csv
 - Export G1 into HTML: Exportuje vizualizáciu G1 do .html
 - Export G2 into HTML: Exportuje vizualizáciu G2 do .html
 - Save analysis: Uloží tabuľky a grafy do jedného PDF reportu 

## Notifikácie a chybové hlásenia
Aplikácia používa tzv. snackbary na zobrazovanie správ, napríklad:
 - Analysis finished – analýza bola úspešne dokončená
 - No file chosen! – nebol vybraný žiadny vstupný súbor
 - Analysis unsuccessful – pri analýze nastala chyba
 - Saved PDF – export do PDF bol úspešný
 - Error during CSV export – export do CSV zly