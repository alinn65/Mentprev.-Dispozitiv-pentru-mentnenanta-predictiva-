#include <mentprev_inferencing.h>

#include <TFT_eSPI.h>
#include <Wire.h>
#include <SparkFun_MMA8452Q.h>
#include <SPI.h>
#include <SD.h>

// =====================================================
// INCLUDE BIBLIOTECA AI EDGE IMPULSE!
// =====================================================

// Wrapper pentru compatibilitate: Edge Impulse se asteapta la tipul
// sensors_event_t din biblioteca Adafruit Unified Sensor, dar noi
// folosim MMA8452Q direct, asa ca definim un struct echivalent minimal.
struct accel_event_t {
  struct { float x, y, z; } acceleration;
};
#define sensors_event_t accel_event_t

// =====================================================
// CONFIGURARE HARDWARE 
// =====================================================
#define PWR_HOLD_PIN     2   // Mentine alimentarea activa (latch MOSFET)
#define DISPLAY_PWR_PIN  4   // Controleaza alimentarea display-ului
#define PWR_BTN_PIN     18   // Buton power / navigare inapoi
#define NAV_BTN_PIN     21   // Buton navigare / confirmare

// SD pe VSPI (al doilea bus SPI al ESP32, separat de cel al display-ului)
#define SD_MOSI     6
#define SD_MISO     7
#define SD_SCK      5
#define SD_CS_PIN   47

// =====================================================
// SETARI PENTRU EDGE IMPULSE
// =====================================================
#define FREQUENCY_HZ        400     // Trebuie sa fie identic cu rata de esantionare din scriptul Python de antrenare
#define INTERVAL_MS         (1000 / (FREQUENCY_HZ + 1))  // +1 pentru a compensa overhead-ul de calcul si a nu depasi frecventa tinta
#define WINDOW_LENGTH_MS    2000    // Lungimea ferestrei de analiza (setata pe platforma Edge Impulse)
#define NUMBER_OF_SAMPLES   (FREQUENCY_HZ * (WINDOW_LENGTH_MS / 1000))  // Total esantioane per rulare: 400 * 2 = 800

// Buffer plat pentru datele accelerometrului inainte de inferenta.
// Ordinea datelor: [X0, Y0, Z0, X1, Y1, Z1, ..., Xn, Yn, Zn]
// Edge Impulse se asteapta exact la aceasta structura (interleaved).
float features[NUMBER_OF_SAMPLES * 3]; // * 3 pentru axele X, Y, Z

// =====================================================
// CONSTANTE UI SI BUTOANE
// =====================================================
const unsigned long DEBOUNCE_MS         = 40;   // Timp de debounce mecanic (ms)
const unsigned long NAV_LONG_MS         = 600;  // Prag pentru apasare lunga pe butonul NAV
const unsigned long PWR_LONG_BACK_MS    = 600;  // Prag pentru apasare lunga pe butonul PWR (inapoi din monitoring)
const float ROTATION_SPEED = 6.0f;              // Viteza animatiei de rotatie a meniului (grade/frame)

// Coordonate display circular de 240x240px
#define SCREEN_CX   120  // Centrul X
#define SCREEN_CY   120  // Centrul Y
#define SCREEN_R    118  // Raza externa (bordura)
#define PLOT_R      105  // Raza pentru grafice interioare
const uint16_t BG_COLOR = TFT_BLACK;

// =====================================================
// TIPURI / STARI
// =====================================================
enum SystemState { MAIN_MENU, FILE_BROWSER, MONITORING };
enum MenuOption  { MENU_START_AI = 0, MENU_VIEW_DATA = 1, MENU_SHUTDOWN = 2, MENU_COUNT = 3 };

// Scorurile returnate de clasificatorul neural (0.0 - 1.0)
float pred_normal = 0.0f;
float pred_dezechilibru = 0.0f;
float pred_blocaj_aer = 0.0f;

// Textul si culoarea afisate in ecranul de monitoring
String curent_diagnostic = "Asteptare date...";
uint16_t current_color = TFT_YELLOW;

// =====================================================
// OBIECTE GLOBALE
// =====================================================
TFT_eSPI tft = TFT_eSPI();
TFT_eSprite sprite = TFT_eSprite(&tft);  // Sprite (buffer off-screen) pentru a elimina flickerul la redesenare
MMA8452Q accel;
SPIClass spiSD(VSPI);  // Bus SPI dedicat cardului SD, diferit de cel al display-ului (HSPI)

char currentLogFile[32] = "/log_ai_1.csv";
unsigned long logIndex = 1;

SystemState state = MAIN_MENU;
MenuOption  selectedOption = MENU_START_AI;

// --- Variabile pentru animatia meniului circular ---
static int logicalMenuPosition = 0;  // Indexul optiunii selectate (0..MENU_COUNT-1)
float currentAngle = 0.0f;           // Unghiul curent al animatiei de rotatie
const float TOP_OFFSET = 180.0f;     // Unghiul de start al primului element in meniu (sus = 180 grade in coordonate arc)

// File Browser
#define MAX_LOG_FILES 50
#define FILES_PER_PAGE 5
char logFileList[MAX_LOG_FILES][32];
int  logFileCount  = 0;
int  browserCursor = 0;  // Pozitia cursorului in lista vizibila
int  browserScroll = 0;  // Offset de scroll (primul fisier vizibil)

// =====================================================
// GESTIONARE BUTOANE CU DEBOUNCE
// =====================================================
struct Button {
  int pin;
  bool lastReading;    // Ultima valoare citita brut de pe pin (poate bouncy)
  bool stableState;    // Starea stabila dupa debounce
  unsigned long lastDebounce;  // Momentul ultimei schimbari de semnal
  unsigned long pressStart;    // Momentul cand butonul a fost apasat
  bool wasPressed;     // Flag: a fost detectata o apasare in curs
  bool longFired;      // Flag: evenimentul de apasare lunga a fost deja emis (evita repetitie)
};

Button navBtn = { NAV_BTN_PIN, HIGH, HIGH, 0, 0, false, false };
Button pwrBtn = { PWR_BTN_PIN, HIGH, HIGH, 0, 0, false, false };

// Actualizeaza starea interna a butonului (debounce).
// Trebuie apelata frecvent din loop sau inainte de a citi starea butonului.
void updateButton(Button& b) {
  bool reading = digitalRead(b.pin);
  if (reading != b.lastReading) b.lastDebounce = millis();
  b.lastReading = reading;
  if ((millis() - b.lastDebounce) > DEBOUNCE_MS) {
    if (reading != b.stableState) {
      b.stableState = reading;
      // Butonul este activ LOW (pull-up intern), deci LOW = apasat
      if (b.stableState == LOW) { b.pressStart = millis(); b.wasPressed = true; b.longFired = false; }
    }
  }
}

// Returneaza true o singura data daca butonul a fost apasat si eliberat rapid (sub threshold ms).
bool clickShort(Button& b, unsigned long threshold) {
  updateButton(b);
  if (b.wasPressed && b.stableState == HIGH && !b.longFired) {
    if (millis() - b.pressStart < threshold) { b.wasPressed = false; return true; }
    b.wasPressed = false;
  }
  return false;
}

// Returneaza true o singura data daca butonul este tinut apasat mai mult de threshold ms.
// longFired previne declansarea repetata cat timp butonul ramane apasat.
bool clickLong(Button& b, unsigned long threshold) {
  updateButton(b);
  if (b.wasPressed && b.stableState == LOW && !b.longFired) {
    if (millis() - b.pressStart >= threshold) { b.longFired = true; return true; }
  }
  return false;
}

// =====================================================
// PROTOTIPURI
// =====================================================
bool initAccelerometer();
bool initDisplay();
bool initSDCard();
void drawCenteredMessage(const char* line1, const char* line2, uint16_t color);
void drawMonitoringUI();
void prepareNextLogFile();
void logAIDataToSD();
void drawMainMenu();
void playMenuTransitionIn();
void playMenuTransitionOut();
void doShutdown();
void drawCircularClipBorder(uint16_t color);
void runAI_Inference();

// =====================================================
// SETUP
// =====================================================
void setup() {
  // Mentinem tensiunea de alimentare activa imediat dupa pornire.
  // Fara aceasta linie, circuitul de latch ar putea opri dispozitivul
  // inainte ca restul initializarilor sa se termine.
  pinMode(PWR_HOLD_PIN, OUTPUT);
  digitalWrite(PWR_HOLD_PIN, HIGH);

  // Display-ul are propriul regulator de tensiune controlat separat
  pinMode(DISPLAY_PWR_PIN, OUTPUT);
  digitalWrite(DISPLAY_PWR_PIN, LOW);  // LOW = pornit (logica inversa pe acest circuit)
  delay(50);

  Serial.begin(115200);
  pinMode(PWR_BTN_PIN, INPUT_PULLUP);
  pinMode(NAV_BTN_PIN, INPUT_PULLUP);

  bool okDisplay = initDisplay();
  bool okSD      = initSDCard();
  bool okAccel   = initAccelerometer();

  // Daca oricare periferic esueaza, nu are sens sa continuam —
  // blocam executia si semnalam eroarea pe serial.
  if (!okDisplay || !okAccel || !okSD) {
     Serial.println("EROARE HARDWARE INIT!");
     while (true) delay(1000);
  }

  prepareNextLogFile();

  playMenuTransitionIn();
  state = MAIN_MENU;
}

// =====================================================
// LOOP MAIN
// =====================================================
void loop() {

  // ---- MAIN MENU ----
  if (state == MAIN_MENU) {
    // PWR BTN = navigare stanga/sus in meniu
    if (clickShort(pwrBtn, NAV_LONG_MS)) {
      logicalMenuPosition--;
      // Wrap circular: daca depasim in jos, sarim la sfarsit
      if (logicalMenuPosition < 0) { logicalMenuPosition += MENU_COUNT; currentAngle += 360.0f; }
      selectedOption = (MenuOption)logicalMenuPosition;
    }
    // NAV BTN = navigare dreapta/jos in meniu
    if (clickShort(navBtn, NAV_LONG_MS)) {
      logicalMenuPosition++;
      // Wrap circular: daca depasim in sus, sarim la inceput
      if (logicalMenuPosition >= MENU_COUNT) { logicalMenuPosition -= MENU_COUNT; currentAngle -= 360.0f; }
      selectedOption = (MenuOption)logicalMenuPosition;
    }

    // Apasare lunga pe NAV = confirmare optiune selectata
    if (clickLong(navBtn, NAV_LONG_MS)) {
      if (selectedOption == MENU_START_AI) {
        playMenuTransitionOut();
        prepareNextLogFile();  // Creeaza un fisier CSV nou pentru aceasta sesiune
        state = MONITORING;
      } else if (selectedOption == MENU_VIEW_DATA) {
         drawCenteredMessage("IN CURAND", "FUNCTIE OPRITA", TFT_YELLOW); 
         delay(1500);
         playMenuTransitionIn();
      } else if (selectedOption == MENU_SHUTDOWN) {
        doShutdown();
      }
    }
    updateButton(pwrBtn);  // Asigura ca pwrBtn este actualizat chiar daca nu a produs un click in aceasta iteratie
    if (state == MAIN_MENU) drawMainMenu();
  }

  // ---- MONITORING (AI INFERENCE) ----
  else if (state == MONITORING) {
    // Apasare lunga pe PWR = iesire din monitoring inapoi la meniu
    if (clickLong(pwrBtn, PWR_LONG_BACK_MS)) {
      playMenuTransitionIn();
      state = MAIN_MENU;
      return;
    }

    // Desenam UI inainte de inferenta, astfel utilizatorul vede starea din
    // ciclul anterior in timp ce se colecteaza noile date (2 secunde)
    drawMonitoringUI();
    
    runAI_Inference();
  }
}

// =====================================================
// ACHIZITIE DATE SI RULARE RETEAUA NEURALA
// =====================================================
void runAI_Inference() {
    
    // Avertizam utilizatorul ca suntem in faza de colectare (blocanta ~2s)
    sprite.setTextColor(TFT_WHITE);
    sprite.drawString("Colectare Date...", SCREEN_CX, 180, 2);
    sprite.pushSprite(0, 0);

    // === PASUL 1: Umplere buffer cu 800 esantioane la 400Hz ===
    // Bucla este blocanta: nu se ruleaza nimic altceva timp de ~2 secunde.
    // Aceasta este o decizie intentionata — intreruperile UI ar distorsiona timing-ul.
    for (int i = 0; i < NUMBER_OF_SAMPLES; i++) {
        unsigned long start_time = millis();
        
        if (accel.available()) {
            accel.read();
            // .cx / .cy / .cz sunt valorile in g calculate de librarie (float)
            // Ordinea trebuie sa fie identica cu cea din scriptul de colectare Python
            features[i * 3 + 0] = accel.cx;
            features[i * 3 + 1] = accel.cy;
            features[i * 3 + 2] = accel.cz;
        }
        // Busy-wait pentru a mentine ritmul de esantionare.
        // delay() nu este suficient de precis la 400Hz.
        while (millis() - start_time < INTERVAL_MS) { }
    }

    // === PASUL 2: Cream semnalul de intrare pentru Edge Impulse ===
    // signal_from_buffer impacheteaza pointerul la bufferul nostru intr-o
    // structura signal_t pe care o intelege pipeline-ul DSP al Edge Impulse.
    signal_t features_signal;
    int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &features_signal);
    if (err != 0) {
        curent_diagnostic = "Eroare Buffer AI";
        current_color = TFT_RED;
        return;
    }

    // === PASUL 3: Rulam clasificatorul neural ===
    // Al treilea parametru (false) dezactiveaza debug output pe serial
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

    if (res != EI_IMPULSE_OK) {
        curent_diagnostic = "Eroare AI";
        current_color = TFT_RED;
        return;
    }

    // === PASUL 4: Interpretam rezultatele ===
    // Parcurgem toate clasele si identificam clasa cu scorul maxim (argmax).
    // Salvam si scorurile individuale pentru barele din UI.
    float max_score = 0;
    int max_index = -1;

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        String label = String(result.classification[i].label);
        float value = result.classification[i].value;

        // Mapam label-urile la variabilele globale de afisare
        if(label == "normal") pred_normal = value;
        else if (label == "dezechilibru") pred_dezechilibru = value;
        else if (label == "blocaj_aer") pred_blocaj_aer = value;

        if (value > max_score) {
            max_score = value;
            max_index = i;
        }
    }

    // === PASUL 5: Actualizam UI pe baza clasei castigatoare ===
    if (max_index != -1) {
        String castigator = String(result.classification[max_index].label);
        
        if (castigator == "normal") {
             curent_diagnostic = "NORMAL";
             current_color = TFT_GREEN;
        } 
        else if (castigator == "dezechilibru") {
             curent_diagnostic = "DEZECHILIBRU";
             current_color = TFT_RED;
        }
        else if (castigator == "blocaj_aer") {
             // TODO: schimba textul in "BLOCAJ AER" pentru consistenta cu celelalte label-uri
             curent_diagnostic = "blocaj_aer";
             current_color = TFT_YELLOW;
        }
    }

    // === PASUL 6: Salvam rezultatul pe SD ===
    logAIDataToSD();
}

// =====================================================
// UI PENTRU MONITORIZARE LIVE AI
// =====================================================
void drawMonitoringUI() {
  sprite.fillSprite(BG_COLOR);
  // Inel decorativ de fundal (gri inchis) — da aspectul unui cadran
  sprite.drawSmoothArc(SCREEN_CX, SCREEN_CY, 112, 104, 0, 360, sprite.color565(28, 28, 28), BG_COLOR);

  sprite.setTextColor(TFT_DARKGREY);
  sprite.drawString("DIAGNOSTIC AI:", SCREEN_CX, 40, 2);

  // Rezultatul curent al clasificatorului, centrat pe ecran
  sprite.setTextColor(current_color);
  sprite.drawString(curent_diagnostic, SCREEN_CX, SCREEN_CY - 15, 4);

  // Bare orizontale proportionale cu scorul fiecarei clase (0..100px)
  int startY = 130;
  
  // Bara "Normal"
  sprite.setTextColor(TFT_GREEN);
  sprite.drawString("NRM", 40, startY, 1);
  sprite.fillRect(60, startY - 4, (int)(pred_normal * 100), 8, TFT_GREEN);
  
  // Bara "Dezechilibru"
  sprite.setTextColor(TFT_RED);
  sprite.drawString("DEZ", 40, startY + 15, 1);
  sprite.fillRect(60, startY + 11, (int)(pred_dezechilibru * 100), 8, TFT_RED);
  
  // Bara "Blocaj Aer" (PAL = prescurtare afisata)
  sprite.setTextColor(TFT_YELLOW);
  sprite.drawString("PAL", 40, startY + 30, 1);
  sprite.fillRect(60, startY + 26, (int)(pred_blocaj_aer * 100), 8, TFT_YELLOW);

  // Redesenam bordura circulara peste sprite pentru a "taia" orice element
  // care a iesit in afara cercului (clipping manual, TFT_eSPI nu are clipping nativ)
  drawCircularClipBorder(sprite.color565(25, 25, 25));
  sprite.pushSprite(0, 0);
}

// =====================================================
// UTILITARE HARDWARE / UI
// =====================================================

// Deseneaza un cerc subtire la marginea display-ului pentru a masca
// artefactele vizuale de la marginea sprite-ului circular.
void drawCircularClipBorder(uint16_t color) {
  sprite.drawSmoothCircle(SCREEN_CX, SCREEN_CY, SCREEN_R - 1, color, BG_COLOR);
}

void playMenuTransitionIn() {
   sprite.fillSprite(BG_COLOR);
   sprite.setTextColor(TFT_CYAN);
   sprite.drawString("SYSTEM READY", SCREEN_CX, SCREEN_CY, 4);
   drawCircularClipBorder(sprite.color565(20, 20, 20));
   sprite.pushSprite(0, 0);
   delay(800);
}

void playMenuTransitionOut() {
    sprite.fillSprite(BG_COLOR);
    sprite.pushSprite(0, 0);
    delay(200);
}

void drawMainMenu() {
  const char* labels[MENU_COUNT] = { "START AI", "VIEW DATA", "SHUTDOWN" };
  const uint16_t colors[MENU_COUNT] = { TFT_CYAN, TFT_YELLOW, TFT_RED };
  sprite.fillSprite(BG_COLOR);
  int sel = (int)selectedOption;
  uint16_t selColor = colors[sel];

  // Afisam doar optiunea selectata — meniu simplu, fara liste
  sprite.setTextColor(selColor);
  sprite.drawString(labels[sel], SCREEN_CX, SCREEN_CY, 4);
  drawCircularClipBorder(sprite.color565(20, 20, 20));
  sprite.pushSprite(0, 0);
}

// Oprire controlata: mai intai inchidem display-ul, apoi eliberam latch-ul de alimentare.
// Ordinea conteaza — daca eliberam latch-ul primul, display-ul ar putea ramane aprins
// cu imagine inghetata pana la descarcarea condensatoarelor.
void doShutdown() {
  digitalWrite(DISPLAY_PWR_PIN, HIGH);  // Opreste display-ul
  delay(10);
  digitalWrite(PWR_HOLD_PIN, LOW);      // Elibereaza latch-ul — tensiunea cade
  while (1);                            // Nu ar trebui sa ajungem aici niciodata
}

void drawCenteredMessage(const char* line1, const char* line2, uint16_t color) {
  sprite.fillSprite(BG_COLOR);
  sprite.setTextColor(TFT_WHITE);
  sprite.drawString(line1, SCREEN_CX, SCREEN_CY - 20, 4);
  sprite.setTextColor(color);
  sprite.drawString(line2, SCREEN_CX, SCREEN_CY + 14, 2);
  drawCircularClipBorder(sprite.color565(25, 25, 25));
  sprite.pushSprite(0, 0);
}

// Cauta urmatorul nume de fisier disponibil pe SD (/log_ai_1.csv, /log_ai_2.csv, ...)
// astfel incat sesiunile anterioare nu sunt suprascrise niciodata.
void prepareNextLogFile() {
  int fileNum = 1;
  while (true) {
    sprintf(currentLogFile, "/log_ai_%d.csv", fileNum);
    if (!SD.exists(currentLogFile)) break;
    fileNum++;
  }
}

void logAIDataToSD() {
  // Deschidem in append daca fisierul exista, altfel il cream de la zero
  File dataFile = SD.open(currentLogFile, SD.exists(currentLogFile) ? FILE_APPEND : FILE_WRITE);
  if (dataFile) {
    // Scriem header-ul CSV doar daca fisierul este gol (prima scriere)
    if (dataFile.size() == 0) dataFile.println("Timp_ms,Diagnostic,Normal_%,Dezechilibru_%,Pale_%");
    dataFile.printf("%lu,%s,%.2f,%.2f,%.2f\n", millis(), curent_diagnostic.c_str(), pred_normal, pred_dezechilibru, pred_blocaj_aer);
    dataFile.close();  // Inchidem explicit la fiecare scriere pentru a evita coruptia datelor la oprire brusca
  }
}

bool initAccelerometer() {
  // SDA=8, SCL=9 — pinii I2C configurati hardware pe acest PCB
  Wire.begin(8, 9);
  delay(100);  // Pauza pentru stabilizarea tensiunii pe linia I2C
  if (!accel.begin(Wire, 0x1C)) return false;  // 0x1C = adresa I2C a MMA8452Q cand SA0=GND
  accel.setScale(SCALE_8G);   // Domeniu de masurare +/-8g (potrivit pentru vibratii puternice)
  accel.setDataRate(ODR_400); // Output Data Rate = 400Hz, identic cu frecventa de antrenare
  return true;
}

bool initDisplay() {
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(BG_COLOR);
  sprite.setColorDepth(8);  // 8-bit culoare pentru a economisi RAM (256 culori suficiente pentru acest UI)
  if (!sprite.createSprite(240, 240)) return false;  // Aloca buffer off-screen 240x240px in PSRAM/heap
  sprite.setTextDatum(MC_DATUM);  // Text centrat atat orizontal cat si vertical fata de coordonatele date
  return true;
}

bool initSDCard() {
  // Initializam bus-ul VSPI cu pinii dedicati cardului SD
  spiSD.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS_PIN);
  // 4MHz clock — valoare conservatoare, sigura pentru majoritatea cardurilor SD
  return SD.begin(SD_CS_PIN, spiSD, 4000000);
}

