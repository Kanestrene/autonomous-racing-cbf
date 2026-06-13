#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UART UUIDs estilo Nordic UART
#define SERVICE_UUID        "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_RX   "6E400002-B5A3-F393-E0A9-E50E24DCCA9E" // write
#define CHARACTERISTIC_TX   "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" // notify

const float v_max = 0.47f;

float v_cmd = 0.0f, d_cmd = 0.0f;
unsigned long lastPacketTime = 0;
const unsigned long timeoutMs = 1000;

unsigned long lastLoopTime = 0;
const unsigned long loopPeriod = 20;

int AIN1 = D1;
int AIN2 = D2;
int STBY = D3;
int BIN1 = D4;
int BIN2 = D5;
int PWMA = D0;
int PWMB = D6;

bool aux = 0;
bool deviceConnected = false;

bool parseCommand(const char* msg, float& v, float& d) {
  return sscanf(msg, "V:%f D:%f", &v, &d) == 2;
}

float grausParaRadianos(float graus) {
  return graus * PI / 180.0f;
}

const float delta_max = grausParaRadianos(13.0f);

int velocity_to_pwm(float v) {
  float ratio = constrain(fabs(v) / v_max, 0.0f, 1.0f);
  return (int)(ratio * 255.0f);
}

int delta_to_pwm(float delta) {
  float ratio = constrain(fabs(delta) / delta_max, 0.0f, 1.0f);
  return (int)(ratio * 255.0f);
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) {
    deviceConnected = true;
    Serial.println("BLE ligado");
  }

  void onDisconnect(BLEServer* server) {
    deviceConnected = false;
    Serial.println("BLE desligado");
    server->startAdvertising();
  }
};

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) {
    String incoming = characteristic->getValue();

    if (incoming.length() > 0) {
      Serial.print("Recebido BLE: ");
      Serial.println(incoming);

      float v, d;
      if (parseCommand(incoming.c_str(), v, d)) {
        v_cmd = v;
        d_cmd = d;
        lastPacketTime = millis();
        analogWrite(PWMB, 120);
      } else {
        Serial.println("Formato invalido");
      }
    }
  }
};

void setupBLE() {
  BLEDevice::init("XIAO-C3-CAR");

  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService* service = server->createService(SERVICE_UUID);

  BLECharacteristic* txCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_TX,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  txCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic* rxCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_RX,
    BLECharacteristic::PROPERTY_WRITE
  );
  rxCharacteristic->setCallbacks(new RxCallbacks());

  service->start();

  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->start();

  Serial.println("BLE pronto. Liga-te a: XIAO-C3-CAR");
}

void setup() {
  Serial.begin(115200);

  pinMode(D0, OUTPUT);
  pinMode(D1, OUTPUT);
  pinMode(D2, OUTPUT);
  pinMode(D3, OUTPUT);
  pinMode(D4, OUTPUT);
  pinMode(D5, OUTPUT);
  pinMode(D6, OUTPUT);

  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);

  lastPacketTime = millis();

  setupBLE();
}

void loop() {
  if (millis() - lastLoopTime < loopPeriod) {
    return;
  }
  lastLoopTime = millis();

  bool timeout = (millis() - lastPacketTime) > timeoutMs;

  if (timeout) {
    analogWrite(PWMB, 0);
    analogWrite(PWMA, 0);
    aux = 0;
    return;
  }

  int pwm_v = velocity_to_pwm(v_cmd);
  int pwm_d = delta_to_pwm(d_cmd);

  if (v_cmd > 0) {
    digitalWrite(STBY, HIGH);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, HIGH);
  } else if (v_cmd < 0) {
    digitalWrite(STBY, HIGH);
    digitalWrite(BIN1, HIGH);
    digitalWrite(BIN2, LOW);
  }

  if (d_cmd > 0.05) {
    digitalWrite(STBY, HIGH);
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
  } else if (d_cmd < -0.05) {
    digitalWrite(STBY, HIGH);
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
  }

  analogWrite(PWMB, pwm_v);
  analogWrite(PWMA, pwm_d);
}