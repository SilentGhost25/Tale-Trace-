#include <WiFi.h>
#include <WebServer.h>
#include <U8g2lib.h>
#include <Wire.h>

// ================= WIFI CREDENTIALS =================
// Fill these in locally. Do NOT commit real credentials to source control.
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ================= STATIC IP CONFIGURATION =================
IPAddress local_IP(192, 168, 1, 26);   // Fixed IP for Button DevKit
IPAddress gateway(192, 168, 1, 1);    // Router IP
IPAddress subnet(255, 255, 255, 0);   // Subnet
IPAddress primaryDNS(8, 8, 8, 8);     // Google DNS

// ================= HARDWARE PINS ====================
const int BTN_MOMENTARY_PIN = 4;  // GPIO 4 connected to Push Button
const int BTN_TOGGLE_PIN    = 5;  // GPIO 5 connected to Toggle Switch

// ================= OLED SETUP (SH1106 I2C) =================
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

String currentText = "TaleTrace Ready!";
String formattedLines[40];    // Holds up to 40 wrapped lines
int totalLines = 0;
int currentLineIndex = 0;      // Controls manual scroll position
const int VISIBLE_LINES = 4;   // Lines displayed per page

// Button State Management & Debounce
bool lastBothState = false;
bool lastMomentaryState = false;
bool lastToggleState = false;
unsigned long lastDebounceTime = 0;
unsigned long lastSerialBroadcast = 0;
const unsigned long DEBOUNCE_DELAY = 150; // 150ms debounce window

WebServer server(8080);

// Format raw string into lines that fit the display width
int prepareTextLines(String text) {
  u8g2.setFont(u8g2_font_profont12_tf);
  const int lineLength = 18; // ~18 characters per line
  String remaining = text;
  int count = 0;

  while (remaining.length() > 0 && count < 40) {
    if (remaining.length() <= lineLength) {
      formattedLines[count++] = remaining;
      break;
    }

    int splitIndex = remaining.lastIndexOf(' ', lineLength);
    if (splitIndex == -1 || splitIndex == 0) {
      splitIndex = lineLength;
    }

    formattedLines[count++] = remaining.substring(0, splitIndex);
    remaining = remaining.substring(splitIndex);
    remaining.trim();
  }

  return count;
}

void renderCurrentPage() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_profont12_tf);

  int startY = 14;
  int lineHeight = 14;

  for (int i = 0; i < VISIBLE_LINES; i++) {
    int lineToDraw = currentLineIndex + i;
    if (lineToDraw < totalLines) {
      u8g2.drawStr(0, startY + (i * lineHeight), formattedLines[lineToDraw].c_str());
    }
  }

  u8g2.sendBuffer();
}

// Update text on OLED and reset view to line 0
void updateOLED(String text) {
  currentText = text;
  totalLines = prepareTextLines(currentText);
  currentLineIndex = 0; // Reset position to top
  renderCurrentPage();
}

// Scroll to next page of lines, or loop back to top
void scrollOLEDNext() {
  if (totalLines <= VISIBLE_LINES) return; // No scroll needed if short text

  currentLineIndex += VISIBLE_LINES;

  // Jump back to start if we exceed total lines
  if (currentLineIndex >= totalLines) {
    currentLineIndex = 0;
  }

  renderCurrentPage();
}

// Check hardware buttons and handle scrolling when both are active
void checkScrollTrigger() {
  bool momentary_state = (digitalRead(BTN_MOMENTARY_PIN) == LOW);
  bool toggle_state    = (digitalRead(BTN_TOGGLE_PIN) == LOW);
  bool currentBothState = (momentary_state && toggle_state);

  // Trigger scroll on transition to BOTH ACTIVE
  if (currentBothState && !lastBothState) {
    if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
      scrollOLEDNext();
      lastDebounceTime = millis();
    }
  }

  lastBothState = currentBothState;
}

String getButtonJson() {
  bool momentary_state = (digitalRead(BTN_MOMENTARY_PIN) == LOW);
  bool toggle_state    = (digitalRead(BTN_TOGGLE_PIN) == LOW);
  bool both_state      = (momentary_state && toggle_state);

  String jsonResponse = "{";
  jsonResponse += "\"btn_momentary\":" + String(momentary_state ? "true" : "false") + ",";
  jsonResponse += "\"btn_toggle\":" + String(toggle_state ? "true" : "false") + ",";
  jsonResponse += "\"both_active\":" + String(both_state ? "true" : "false");
  jsonResponse += "}";
  return jsonResponse;
}

// GET /buttons -> Returns JSON button states
void handleButtons() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", getButtonJson());
}

// POST /display -> Receives backend text definitions
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

void handleSerialCommunication() {
  // 1. Check for incoming Serial commands (e.g. DISPLAY:<text>)
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.startsWith("DISPLAY:")) {
      String textToDisplay = command.substring(8);
      updateOLED(textToDisplay);
      Serial.println("OK:DISPLAY_UPDATED");
    } else if (command == "POLL") {
      Serial.println(getButtonJson());
    } else if (command == "SCROLL") {
      scrollOLEDNext();
      Serial.println("OK:SCROLLED");
    }
  }

  // 2. Broadcast state changes or periodic heartbeat over Serial
  bool m = (digitalRead(BTN_MOMENTARY_PIN) == LOW);
  bool t = (digitalRead(BTN_TOGGLE_PIN) == LOW);

  unsigned long now = millis();
  if (m != lastMomentaryState || t != lastToggleState || (now - lastSerialBroadcast > 200)) {
    Serial.println(getButtonJson());
    lastMomentaryState = m;
    lastToggleState = t;
    lastSerialBroadcast = now;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  u8g2.begin();
  updateOLED("TaleTrace Ready!");

  pinMode(BTN_MOMENTARY_PIN, INPUT_PULLUP);
  pinMode(BTN_TOGGLE_PIN, INPUT_PULLUP);

  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS)) {
    Serial.println("Static IP Config Failed; using DHCP");
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 10) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nDevKit WiFi Connected!");
    server.on("/buttons", HTTP_GET, handleButtons);
    server.on("/display", HTTP_POST, handleDisplay);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("DevKit Server listening on Port 8080.");
  } else {
    Serial.println("\nWiFi not connected; running in USB Serial Mode.");
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }
  checkScrollTrigger();
  handleSerialCommunication();
}
