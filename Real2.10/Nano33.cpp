#include <Arduino.h>
#include <SPI.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include <stdlib.h>
#include <string.h>

const char* WIFI_SSID = "nome_da_tua_rede";
const char* WIFI_PASS = "palavra_passe";

WiFiUDP udp;
const unsigned int UDP_PORT = 5005;

const float v_max = 0.47f;
const float MAX_VOLTAGE = 3.3f;
const float delta_max = 16.5f * PI / 180.0f;

float v_cmd = 0.0f;
float d_cmd = 0.0f;

// Usa pinos PWM validos da Nano 33 IoT.
// Ajusta estes pinos conforme a tua ligacao fisica.
const int VelocityPin = 9;
const int DeltaPin = 10;
const int pwmResolution = 8;  // 0 a 255

void connectWifi() {
    if (WiFi.status() == WL_NO_MODULE) {
        Serial.println("Modulo WiFi nao encontrado na Nano 33.");
        while (true) {
            delay(1000);
        }
    }

    int status = WL_IDLE_STATUS;
    Serial.print("A ligar ao WiFi");

    while (status != WL_CONNECTED) {
        status = WiFi.begin(WIFI_SSID, WIFI_PASS);
        delay(5000);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi ligado");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

void skipSpaces(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
        ++p;
    }
}

bool parseTaggedFloat(const char*& p, const char* tag, float& value) {
    skipSpaces(p);

    const size_t tagLen = strlen(tag);
    if (strncmp(p, tag, tagLen) != 0) {
        return false;
    }

    p += tagLen;

    char* endPtr = nullptr;
    value = strtof(p, &endPtr);
    if (endPtr == p) {
        return false;
    }

    p = endPtr;
    return true;
}

bool parseCommand(const char* msg, float& v, float& d) {
    const char* p = msg;

    if (!parseTaggedFloat(p, "V:", v)) {
        return false;
    }

    if (!parseTaggedFloat(p, "D:", d)) {
        return false;
    }

    skipSpaces(p);
    return *p == '\0';
}

float velocityToVoltage(float v) {
    float ratio = constrain(v / v_max, 0.0f, 1.0f);
    return ratio * MAX_VOLTAGE;
}

float deltaToVoltage(float delta) {
    float ratio = constrain(delta / delta_max, -1.0f, 1.0f);
    return (ratio * 0.5f + 0.5f) * MAX_VOLTAGE;
}

int voltageToPwm(float voltage) {
    const int pwmMax = (1 << pwmResolution) - 1;
    voltage = constrain(voltage, 0.0f, MAX_VOLTAGE);
    return (int)((voltage / MAX_VOLTAGE) * pwmMax + 0.5f);
}

void writeOutputs(float v, float delta) {
    analogWrite(VelocityPin, voltageToPwm(velocityToVoltage(v)));
    analogWrite(DeltaPin, voltageToPwm(deltaToVoltage(delta)));
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
    }

    pinMode(VelocityPin, OUTPUT);
    pinMode(DeltaPin, OUTPUT);
    analogWriteResolution(pwmResolution);

    // Na direcao, 0 rad fica a meio da escala PWM.
    writeOutputs(0.0f, 0.0f);

    connectWifi();

    if (udp.begin(UDP_PORT)) {
        Serial.print("A escutar UDP na porta ");
        Serial.println(UDP_PORT);
    } else {
        Serial.println("Erro a iniciar UDP");
    }
}

void loop() {
    int packetSize = udp.parsePacket();

    if (!packetSize) {
        return;
    }

    char incoming[128];
    int len = udp.read(incoming, sizeof(incoming) - 1);

    if (len <= 0) {
        return;
    }

    incoming[len] = '\0';

    Serial.print("Recebido: ");
    Serial.println(incoming);

    float v = 0.0f;
    float d = 0.0f;

    if (parseCommand(incoming, v, d)) {
        v_cmd = v;
        d_cmd = d;

        Serial.print("Velocidade: ");
        Serial.println(v_cmd);

        Serial.print("Delta: ");
        Serial.println(d_cmd);

        writeOutputs(v_cmd, d_cmd);
    } else {
        Serial.println("Formato invalido");
    }
}
