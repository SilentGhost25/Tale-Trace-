// =====================================================================
//  TaleTrace — ESP32 Button + OLED DevKit
//  ------------------------------------------------------------------
//  Pins:  GPIO 4  -> momentary push button  (active LOW)
//         GPIO 5  -> toggle switch          (active LOW)
//         GPIO 23 -> intent push button     (active LOW, momentary)
//         GPIO 21 -> OLED SDA
//         GPIO 22 -> OLED SCL
//
//  Each switch:  GPIO ---- switch ---- GND   (internal pull-up)
// =====================================================================

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <U8g2lib.h>

// =====================================================================
//  OLED DRIVER — SH1106 (hardware default) or SSD1306
// =====================================================================
// #define USE_SSD1306
#define USE_SH1106

#if defined(USE_SSD1306)
  U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#elif defined(USE_SH1106)
  U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
#else
  #error "Pick SSD1306 or SH1106"
#endif

// =====================================================================
//  WIFI CREDENTIALS  — EDIT
// =====================================================================
const char* WIFI_SSID     = "Tarun";
const char* WIFI_PASSWORD = "12345678";

// =====================================================================
//  STATIC IP (optional)
// =====================================================================
#define USE_STATIC_IP 0
IPAddress local_IP(192, 168, 1, 26);
IPAddress gateway (192, 168, 1, 1);
IPAddress subnet  (255, 255, 255, 0);
IPAddress dns1    (8, 8, 8, 8);

// =====================================================================
//  HARDWARE PINS
// =====================================================================
const int BTN_MOMENTARY_PIN = 4;
const int BTN_TOGGLE_PIN    = 5;
const int BTN_INTENT_PIN    = 23;   // change to 25 if this pin is dead

// =====================================================================
//  OLED STATE
// =====================================================================
String currentText = "TaleTrace Ready!";
const int MAX_LINES = 40;
String formattedLines[MAX_LINES];
int totalLines = 0;
int currentLineIndex = 0;
const int VISIBLE_LINES = 4;

// =====================================================================
//  BUTTON STATE
// =====================================================================
bool lastBothState = false;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 150;
unsigned long lastAutoScrollMs = 0;
const unsigned long AUTO_SCROLL_INTERVAL_MS = 1500;

// Intent tap latches
bool intentLevel        = false;
bool intentPressLatch   = false;
bool intentReleaseLatch = false;
bool lastIntentReading  = false;
unsigned long lastIntentChange = 0;
const unsigned long INTENT_DEBOUNCE_MS = 30;

// Serial heartbeat
unsigned long lastSerialHeartbeat = 0;
const unsigned long SERIAL_HEARTBEAT_MS = 500;

WebServer server(8080);

// =====================================================================
//  OLED HELPERS
// =====================================================================
void oledShow(const String &text) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.drawStr(0, 14, text.c_str());
  u8g2.sendBuffer();
}

int prepareTextLines(const String &src) {
  for (int i = 0; i < MAX_LINES; i++) formattedLines[i] = "";

  String text = src;
  text.replace("\r", "");

  const int maxChars = 20;
  int count = 0;
  int startIdx = 0;
  int textLen = text.length();

  while (startIdx < textLen && count < MAX_LINES) {
    int newlineIdx = text.indexOf('\n', startIdx);
    int chunkEnd;

    if (newlineIdx != -1 && (newlineIdx - startIdx) <= maxChars) {
      chunkEnd = newlineIdx;
    } else {
      chunkEnd = startIdx + maxChars;
      if (chunkEnd > textLen) chunkEnd = textLen;

      if (chunkEnd < textLen) {
        int lastSpace = text.lastIndexOf(' ', chunkEnd);
        if (lastSpace > startIdx) chunkEnd = lastSpace;
      }
    }

    String line = text.substring(startIdx, chunkEnd);
    line.trim();
    if (line.length() > 0) formattedLines[count++] = line;

    startIdx = chunkEnd;
    while (startIdx < textLen &&
           (text.charAt(startIdx) == ' ' || text.charAt(startIdx) == '\n')) {
      startIdx++;
    }
  }
  return count;
}

void renderCurrentPage() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x12_tf);

  int startY = 13;
  int lineHeight = 13;

  for (int i = 0; i < VISIBLE_LINES; i++) {
    int lineToDraw = currentLineIndex + i;
    if (lineToDraw < totalLines) {
      u8g2.drawStr(0, startY + (i * lineHeight), formattedLines[lineToDraw].c_str());
    }
  }
  u8g2.sendBuffer();
}

void updateOLED(const String &text) {
  currentText = text;
  totalLines = prepareTextLines(currentText);
  currentLineIndex = 0;
  renderCurrentPage();
}

void scrollOLEDNext() {
  if (totalLines <= VISIBLE_LINES) return;
  currentLineIndex += VISIBLE_LINES;
  if (currentLineIndex >= totalLines) currentLineIndex = 0;
  renderCurrentPage();
}

// =====================================================================
//  BUTTON HELPERS
// =====================================================================
bool readMomentary() { return digitalRead(BTN_MOMENTARY_PIN) == LOW; }
bool readToggle()    { return digitalRead(BTN_TOGGLE_PIN)    == LOW; }

// ---------------------------------------------------------------
//  Intent edge detection — prints IMMEDIATELY on every tap
// ---------------------------------------------------------------
void updateIntentEdges() {
  bool reading = (digitalRead(BTN_INTENT_PIN) == LOW);
  unsigned long now = millis();

  if (reading != lastIntentReading) {
    lastIntentChange = now;
    lastIntentReading = reading;
  }

  if ((now - lastIntentChange) > INTENT_DEBOUNCE_MS && reading != intentLevel) {
    if (reading) {
      intentPressLatch = true;
      Serial.println(">>> INTENT PRESSED <<<");
    } else {
      intentReleaseLatch = true;
      Serial.println(">>> INTENT RELEASED <<<");
    }
    intentLevel = reading;
  }
}

void checkScrollTrigger() {
  bool momentary_state = readMomentary();
  bool toggle_state    = readToggle();
  bool currentBothState = momentary_state && toggle_state;

  if (currentBothState && !lastBothState) {
    if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
      scrollOLEDNext();
      lastDebounceTime = millis();
      lastAutoScrollMs = millis();
    }
  }

  // Continuous auto-scroll while both buttons are held
  if (currentBothState && (millis() - lastAutoScrollMs > AUTO_SCROLL_INTERVAL_MS)) {
    scrollOLEDNext();
    lastAutoScrollMs = millis();
  }

  lastBothState = currentBothState;
}

// =====================================================================
//  JSON STATE
// =====================================================================
const char* currentMode() {
  bool m = readMomentary();
  bool t = readToggle();
  if (m && t)  return "SCROLL_MODE";
  if (m && !t) return "UPDATE_POSITION";
  if (!m && t) return "MEANING_MODE";
  return "IDLE_READING";
}

String getButtonJson(bool consumeLatches = true) {
  bool m = readMomentary();
  bool t = readToggle();
  bool both = m && t;
  bool intentNow = intentLevel;

  bool pressed  = intentPressLatch;
  bool released = intentReleaseLatch;

  if (consumeLatches) {
    intentPressLatch   = false;
    intentReleaseLatch = false;
  }

  String j = "{";
  j += "\"btn_momentary\":" + String(m ? "true" : "false") + ",";
  j += "\"btn_toggle\":"    + String(t ? "true" : "false") + ",";
  j += "\"both_active\":"   + String(both ? "true" : "false") + ",";
  j += "\"intend\":"         + String(intentNow ? "true" : "false") + ",";
  j += "\"intend_pressed\":" + String(pressed  ? "true" : "false") + ",";
  j += "\"intend_released\":"+ String(released ? "true" : "false") + ",";
  j += "\"mode\":\"" + String(currentMode()) + "\"";
  j += "}";
  return j;
}

// =====================================================================
//  HTTP HANDLERS
// =====================================================================
void handleButtons() {
  String body = getButtonJson(true);
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", body);
}

void handleDisplay() {
  String bodyText = "";
  if (server.hasArg("plain")) {
    bodyText = server.arg("plain");
  } else if (server.args() > 0) {
    bodyText = server.argName(0);
  }

  if (bodyText.length() > 0) {
    updateOLED(bodyText);
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "text/plain", "Display updated successfully");
  } else {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(400, "text/plain", "Bad Request: Empty Body");
  }
}

void handleNotFound() {
  server.send(404, "text/plain", "Not Found");
}

// =====================================================================
//  SERIAL COMMANDS + HEARTBEAT
// =====================================================================
void handleSerial() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("DISPLAY:")) {
      updateOLED(cmd.substring(8));
      Serial.println("OK:DISPLAY_UPDATED");
    } else if (cmd == "POLL") {
      Serial.println(getButtonJson(true));
    } else if (cmd == "SCROLL") {
      scrollOLEDNext();
      Serial.println("OK:SCROLLED");
    } else if (cmd == "I2C") {
      Serial.println("--- I2C SCAN ---");
      for (uint8_t a = 1; a < 127; a++) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
          Serial.printf("  device at 0x%02X\n", a);
        }
      }
      Serial.println("--- end ---");
    }
  }

  unsigned long now = millis();
  if (now - lastSerialHeartbeat > SERIAL_HEARTBEAT_MS) {
    Serial.println(getButtonJson(true));
    lastSerialHeartbeat = now;
  }
}

// =====================================================================
//  SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" TaleTrace ESP32 booting...");
  Serial.println("========================================");

  // --- Buttons ---
  pinMode(BTN_MOMENTARY_PIN, INPUT_PULLUP);
  pinMode(BTN_TOGGLE_PIN,    INPUT_PULLUP);
  pinMode(BTN_INTENT_PIN,    INPUT_PULLUP);

  Serial.printf("Buttons initialized (GPIO %d, %d, %d)\n",
                BTN_MOMENTARY_PIN, BTN_TOGGLE_PIN, BTN_INTENT_PIN);

  // --- I2C + OLED ---
  Wire.begin(21, 22);
  Wire.setClock(400000);

  Serial.println("Scanning I2C bus (SDA=21, SCL=22)...");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  I2C device at 0x%02X\n", a);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("  !! No I2C devices found — check SDA/SCL/VCC/GND !!");
  }

  u8g2.begin();
  oledShow("TaleTrace Ready!");
  Serial.println("OLED initialized.");

  // --- WiFi ---
#if USE_STATIC_IP
  if (!WiFi.config(local_IP, gateway, subnet, dns1)) {
    Serial.println("Static IP config rejected; will use DHCP.");
  } else {
    Serial.println("Static IP configured.");
  }
#else
  Serial.println("Using DHCP.");
#endif

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi \"%s\"", WIFI_SSID);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi CONNECTED");
    Serial.print("  IP:      "); Serial.println(WiFi.localIP());
    Serial.print("  Gateway: "); Serial.println(WiFi.gatewayIP());
    Serial.print("  Subnet:  "); Serial.println(WiFi.subnetMask());
    Serial.print("  RSSI:    "); Serial.println(WiFi.RSSI());

    server.on("/buttons", HTTP_GET,  handleButtons);
    server.on("/display", HTTP_POST, handleDisplay);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("HTTP server listening on port 8080");
  } else {
    Serial.println("WiFi NOT CONNECTED — USB Serial mode only.");
    Serial.println("  Serial cmds: DISPLAY:<text> | POLL | SCROLL | I2C");
  }

  Serial.println("========================================");
  Serial.println("Boot complete. Watch for >>> INTENT lines when you tap the button.");
  Serial.println("========================================");
}

// =====================================================================
//  LOOP
// =====================================================================
void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }

  updateIntentEdges();     // prints >>> INTENT PRESSED/RELEASED <<< immediately
  checkScrollTrigger();
  handleSerial();
}
