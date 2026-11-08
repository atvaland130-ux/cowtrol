/*
  Cowtrol Systems - Arduino main sketch (SHT45 x6 via TCA9548A mux), talks to the Raspberry Pi over USB.
  Replaces the DHT20 version. Reads up to 6 SHT45 sensors behind a TCA9548A I2C multiplexer and reports
  each as a numbered sensor the Pi maps to a quadrant (config.yaml -> quadrant_sensors).

  Hardware:
    - TCA9548A mux: VIN -> 5V, GND -> GND, SDA -> SDA, SCL -> SCL, A0-A2 -> GND (address 0x70)
    - SHT45 on each used mux channel: sensor 1 -> SD0/SC0, sensor 2 -> SD1/SC1, ... sensor 6 -> SD5/SC5
    - One PWM AC dimmer per fan: PWM -> fan pin.  fan 1 -> 9, fan 2 -> 10, fan 3 -> 11, fan 4 -> 3
    (With 2 dimmers: left on pin 9, right on pin 11; pins 10 and 3 just drive nothing.)

  Protocol (9600 baud) - unchanged, so the Pi controller needs no change:
    Arduino -> Pi:  S,<sensor>,<temp_c>,<rh_pct>   one line per working sensor, every ~2 s
                    E,<message>                    all sensors failed / failsafe
                    A,<duty1>,...,<duty4>          acknowledges a fan command (duty actually applied)
    Pi -> Arduino:  F,<duty1>[,<duty2>,...]        fan power in % (0-100); missing fans = 0

  Notes:
    - A sensor that fails to read is simply not reported; the Pi marks it silent and borrows a same-side
      average. With fewer than 6 sensors wired, set config quadrant_sensors so unused ids map nowhere.
    - Any fan request from 1 up to MIN_DUTY runs at MIN_DUTY so the fan never stalls.
    - Failsafe: no fan command for FAILSAFE_MS -> fans go to FAILSAFE_DUTY.
    - You can test by hand in Serial Monitor (line ending "Newline"): type F,50

  No extra libraries needed.
*/
#include <Wire.h>

const uint8_t  MUX_ADDR        = 0x70;     // TCA9548A with A0-A2 at GND
const uint8_t  SHT45_ADDR      = 0x44;     // 0x45 on -B variants
const uint8_t  CMD_MEASURE_HIGH = 0xFD;    // high-precision measurement, heater off
const uint8_t  NUM_SENSORS     = 6;        // mux channels 0..5 -> reported as sensor 1..6

const uint8_t  NUM_FANS        = 4;
const uint8_t  FAN_PINS[NUM_FANS] = {9, 10, 11, 3};   // fan 1, 2, 3, 4
const int      MIN_DUTY        = 35;       // % - real fan cut-in (must match min_duty_pct in config.yaml)
const int      FAILSAFE_DUTY   = 100;      // % - used if the Pi goes silent
const unsigned long FAILSAFE_MS = 120000UL;
const unsigned long READ_MS    = 2000;

unsigned long lastRead = 0, lastCmd = 0;
bool failsafeActive = false;
String buf;

// Sensirion CRC-8: polynomial 0x31, init 0xFF
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

// Only one SHT45 (shared address) can be live at a time, so open one mux channel.
bool selectChannel(uint8_t channel) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write((uint8_t)(1 << channel));
  return Wire.endTransmission() == 0;
}

// Returns true and fills tC/rh on a good reading.
bool readSHT45(float &tC, float &rh) {
  Wire.beginTransmission(SHT45_ADDR);
  Wire.write(CMD_MEASURE_HIGH);
  if (Wire.endTransmission() != 0) return false;

  delay(10);  // max measurement time ~8.3 ms

  if (Wire.requestFrom(SHT45_ADDR, (uint8_t)6) != 6) return false;
  uint8_t d[6];
  for (uint8_t i = 0; i < 6; i++) d[i] = Wire.read();
  if (crc8(d, 2) != d[2] || crc8(d + 3, 2) != d[5]) return false;

  uint16_t rawT  = ((uint16_t)d[0] << 8) | d[1];
  uint16_t rawRH = ((uint16_t)d[3] << 8) | d[4];
  tC = -45.0 + 175.0 * rawT / 65535.0;
  rh = -6.0 + 125.0 * rawRH / 65535.0;
  if (rh < 0) rh = 0; else if (rh > 100) rh = 100;
  return true;
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
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(25000, true);           // a loose wire can't freeze the sketch
#endif
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

  // sensors: report each working SHT45 as its own numbered sensor
  if (millis() - lastRead >= READ_MS) {
    lastRead = millis();
    uint8_t ok = 0;
    for (uint8_t ch = 0; ch < NUM_SENSORS; ch++) {
      float tC, rh;
      if (selectChannel(ch) && readSHT45(tC, rh)) {
        ok++;
        Serial.print(F("S,"));
        Serial.print(ch + 1);                 // sensor id 1..6
        Serial.print(F(","));
        Serial.print(tC, 1);
        Serial.print(F(","));
        Serial.println(rh, 1);
      }
      // a failed channel is left unreported; the Pi marks it silent and borrows a same-side average
    }
    if (ok == 0) Serial.println(F("E,no_sensors_answering"));
  }

  // failsafe
  if (!failsafeActive && millis() - lastCmd > FAILSAFE_MS) {
    for (uint8_t i = 0; i < NUM_FANS; i++) setDuty(FAN_PINS[i], FAILSAFE_DUTY);
    failsafeActive = true;
    Serial.println(F("E,failsafe_no_command"));
  }
}
