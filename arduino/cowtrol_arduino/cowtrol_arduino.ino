/*
  Cowtrol Systems - Arduino Uno R4 WiFi (main sketch, talks to the Raspberry Pi over USB)

  Hardware:
    - Sensor Kit Temperature & Humidity module (DHT20, I2C) through the Sensor Kit middle board
    - One PWM AC dimmer module per fan: VCC -> 5V, GND -> GND, PWM -> the fan's pin
        fan 1 -> pin 9,  fan 2 -> pin 10,  fan 3 -> pin 11,  fan 4 -> pin 3

  Protocol (9600 baud):
    Arduino -> Pi:  S,<temp_c>,<rh_pct>     every 2 s
                    E,<message>             sensor error or failsafe
                    A,<duty1>,...,<duty4>   acknowledges a fan command (duty actually applied)
    Pi -> Arduino:  F,<duty1>[,<duty2>,...] fan power in % (0-100), one value per fan; missing fans = 0

  Fan rules:
    - 0 = off. Any request from 1 up to MIN_DUTY runs at MIN_DUTY, so the fan never stalls.
    - Failsafe: no fan command for FAILSAFE_MS -> fans go to FAILSAFE_DUTY.

  No extra libraries needed. You can test by hand in Serial Monitor (line ending "Newline"): type F,50
*/
#include <Wire.h>

const uint8_t DHT20_ADDR = 0x38;
const uint8_t NUM_FANS = 4;
const uint8_t FAN_PINS[NUM_FANS] = {9, 10, 11, 3};   // fan 1, 2, 3, 4 (all PWM pins on the Uno R4)
const int MIN_DUTY = 35;                       // % - lowest power at which the fan starts from a stop
const int FAILSAFE_DUTY = 100;                // % - used if the Pi goes silent
const unsigned long FAILSAFE_MS = 120000UL;   // 2 min without a Pi command
const unsigned long READ_MS = 2000;

unsigned long lastRead = 0, lastCmd = 0;
bool failsafeActive = false;
String buf;

uint8_t crc8(const uint8_t *d, uint8_t n) {
  uint8_t crc = 0xFF;
  for (uint8_t i = 0; i < n; i++) {
    crc ^= d[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

// Returns 0 = OK, 1 = sensor not answering (wiring), 2 = bad data
int readDHT20(float &tC, float &rh) {
  Wire.beginTransmission(DHT20_ADDR);
  Wire.write((uint8_t)0xAC);
  Wire.write((uint8_t)0x33);
  Wire.write((uint8_t)0x00);
  if (Wire.endTransmission() != 0) return 1;
  delay(80);

  uint8_t d[7];
  if (Wire.requestFrom((uint8_t)DHT20_ADDR, (size_t)7) != 7) return 1;
  for (uint8_t i = 0; i < 7; i++) d[i] = Wire.read();

  if (d[0] & 0x80) return 2;
  if (crc8(d, 6) != d[6]) return 2;

  uint32_t rawH = ((uint32_t)d[1] << 12) | ((uint32_t)d[2] << 4) | (d[3] >> 4);
  uint32_t rawT = ((uint32_t)(d[3] & 0x0F) << 16) | ((uint32_t)d[4] << 8) | d[5];
  rh = rawH * 100.0 / 1048576.0;
  tC = rawT * 200.0 / 1048576.0 - 50.0;
  return 0;
}

// Applies the fan rules and returns the duty actually set
int setDuty(uint8_t pin, int pct) {
  pct = constrain(pct, 0, 100);
  if (pct > 0 && pct < MIN_DUTY) pct = MIN_DUTY;
  analogWrite(pin, map(pct, 0, 100, 0, 255));
  return pct;
}

void handleLine(const String &line) {
  if (!line.startsWith("F,")) return;
  int duty[NUM_FANS];
  int start = 2;
  for (uint8_t i = 0; i < NUM_FANS; i++) {
    duty[i] = 0;                                   // fans not named in the command are switched off
    if (start < 0) continue;
    int c = line.indexOf(',', start);
    duty[i] = (c < 0) ? line.substring(start).toInt() : line.substring(start, c).toInt();
    start = (c < 0) ? -1 : c + 1;
  }
  Serial.print(F("A"));
  for (uint8_t i = 0; i < NUM_FANS; i++) {
    Serial.print(F(","));
    Serial.print(setDuty(FAN_PINS[i], duty[i]));
  }
  Serial.println();
  lastCmd = millis();
  failsafeActive = false;
}

void setup() {
  for (uint8_t i = 0; i < NUM_FANS; i++) {
    pinMode(FAN_PINS[i], OUTPUT);
    digitalWrite(FAN_PINS[i], LOW);           // fans off at power-up
  }
  Serial.begin(9600);
  Wire.begin();
  delay(200);                                 // sensor power-up time
  lastCmd = millis();
}

void loop() {
  // commands from the Pi
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      handleLine(buf);
      buf = "";
    } else if (ch != '\r' && buf.length() < 32) {
      buf += ch;
    }
  }

  // sensor
  if (millis() - lastRead >= READ_MS) {
    lastRead = millis();
    float tC, rh;
    int err = readDHT20(tC, rh);
    if (err == 1) {
      Serial.println(F("E,sensor_not_answering"));
    } else if (err == 2) {
      Serial.println(F("E,sensor_bad_reading"));
    } else {
      Serial.print(F("S,"));
      Serial.print(tC, 1);
      Serial.print(F(","));
      Serial.println(rh, 1);
    }
  }

  // failsafe
  if (!failsafeActive && millis() - lastCmd > FAILSAFE_MS) {
    for (uint8_t i = 0; i < NUM_FANS; i++) setDuty(FAN_PINS[i], FAILSAFE_DUTY);
    failsafeActive = true;
    Serial.println(F("E,failsafe_no_command"));
  }
}
