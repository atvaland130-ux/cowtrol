/*
  SHT45 + TCA9548A mux TEST sketch  (diagnostic only - no fans, no Pi).
 
  Open the Serial Monitor at 115200 baud, line ending doesn't matter.
  Every 2 s it:
    1. Checks the TCA9548A mux responds at 0x70.
    2. For each mux channel 0..5: selects it, scans I2C to show what's there,
       then tries to read the SHT45 at 0x44 and prints Temp / RH / THI.
  A channel with a loose wire just shows "-- no SHT45 --"; the rest keep working.
 
  Wiring:
    TCA9548A: VIN->5V, GND->GND, SDA->SDA(A4), SCL->SCL(A5), A0-A2->GND (addr 0x70)
    SHT45 #n on mux channel n: sensor1->SD0/SC0, sensor2->SD1/SC1, ... sensor6->SD5/SC5
  No libraries needed.
*/
#include <Wire.h>
 
const uint8_t MUX_ADDR   = 0x70;   // TCA9548A, A0-A2 at GND
const uint8_t SHT45_ADDR = 0x44;   // 0x45 on -B variants
const uint8_t CMD_MEAS   = 0xFD;   // high-precision, heater off
const uint8_t NUM_CH     = 6;      // number of sensors used
const uint8_t SENSOR_CH[NUM_CH] = {0, 1, 2, 5, 6, 7};   // physical mux channels in use, in sensor order
 
// Sensirion CRC-8: poly 0x31, init 0xFF
uint8_t crc8(const uint8_t *d, uint8_t n) {
  uint8_t crc = 0xFF;
  for (uint8_t i = 0; i < n; i++) {
    crc ^= d[i];
    for (uint8_t b = 0; b < 8; b++)
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
  }
  return crc;
}
 
bool selectChannel(uint8_t ch) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write((uint8_t)(1 << ch));
  return Wire.endTransmission() == 0;
}
 
// Reads the SHT45 on the currently-selected channel.
bool readSHT45(float &tC, float &rh) {
  Wire.beginTransmission(SHT45_ADDR);
  Wire.write(CMD_MEAS);
  if (Wire.endTransmission() != 0) return false;
  delay(10);                                  // measurement ~8.3 ms
  if (Wire.requestFrom(SHT45_ADDR, (uint8_t)6) != 6) return false;
  uint8_t d[6];
  for (uint8_t i = 0; i < 6; i++) d[i] = Wire.read();
  if (crc8(d, 2) != d[2] || crc8(d + 3, 2) != d[5]) return false;   // bad CRC = noisy wire
  uint16_t rawT  = ((uint16_t)d[0] << 8) | d[1];
  uint16_t rawRH = ((uint16_t)d[3] << 8) | d[4];
  tC = -45.0 + 175.0 * rawT / 65535.0;
  rh = -6.0 + 125.0 * rawRH / 65535.0;
  if (rh < 0) rh = 0; else if (rh > 100) rh = 100;
  return true;
}
 
// NRC THI (same as the controller uses), for a quick sanity number.
float thi(float tC, float rh) {
  return (1.8 * tC + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * tC - 26.0);
}
 
void setup() {
  Serial.begin(115200);
  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(25000, true);           // a loose wire can't freeze the sketch
#endif
  delay(200);
  Serial.println(F("\nSHT45 + TCA9548A test. Reading every 2 s...\n"));
}
 
void loop() {
  // 1. is the mux there?
  Wire.beginTransmission(MUX_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println(F("MUX NOT FOUND at 0x70  (check VIN/GND/SDA/SCL and A0-A2->GND)"));
    delay(2000);
    return;
  }
 
  uint8_t ok = 0;
  for (uint8_t i = 0; i < NUM_CH; i++) {
    uint8_t ch = SENSOR_CH[i];
    Serial.print(F("Ch ")); Serial.print(ch);
    Serial.print(F(" (sensor ")); Serial.print(i + 1); Serial.print(F("): "));
 
    if (!selectChannel(ch)) { Serial.println(F("mux select failed")); continue; }
 
    float tC, rh;
    if (readSHT45(tC, rh)) {
      ok++;
      Serial.print(F("T=")); Serial.print(tC, 2); Serial.print(F(" C  "));
      Serial.print(F("RH=")); Serial.print(rh, 2); Serial.print(F(" %  "));
      Serial.print(F("THI=")); Serial.println(thi(tC, rh), 1);
    } else {
      Serial.println(F("-- no SHT45 --  (loose wire / wrong channel / bad CRC)"));
    }
  }
  Serial.print(F(">> ")); Serial.print(ok);
  Serial.println(F(" of 6 sensors answered\n"));
 
  selectChannel(8);   // deselect all channels (1<<8 = 0 -> none)
  delay(2000);
}
 
