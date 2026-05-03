#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID = "Lopes";
const char* WIFI_PASS = "12345678";

WiFiUDP udp;
const int UDP_PORT = 5005;

const float v_max = 0.47f;

float v_cmd = 0.0f, d_cmd = 0.0f;
unsigned long loopCounter = 0;

int AIN1=D1;
int AIN2=D2;
int STBY=D3;
int BIN1=D4;
int BIN2=D5;
int PWMA=D0;
int PWMB=D6;

void connectWifi() {
    WiFi.mode(WIFI_STA);          
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("A ligar ao WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi ligado");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

bool parseCommand(const char* msg, float& v, float& d) {
    return sscanf(msg, "V:%f D:%f", &v, &d) == 2;
}

float grausParaRadianos(float graus) {
    return graus * PI / 180.0f;
}

const float delta_max = grausParaRadianos(16.5f);

int velocity_to_pwm(float v) {
    float ratio = constrain(fabs(v) / v_max, 0.0f, 1.0f);
    return (int)(ratio * 255.0f);
}

int delta_to_pwm(float delta) {
    float ratio = constrain(fabs(delta) / delta_max, 0.0f, 1.0f);
    return (int)(ratio * 255.0f);
}

void setup() {
    Serial.begin(115200);

    pinMode(D0,OUTPUT);
    pinMode(D1,OUTPUT);
    pinMode(D2,OUTPUT);
    pinMode(D3,OUTPUT);
    pinMode(D4,OUTPUT);
    pinMode(D5,OUTPUT);
    pinMode(D6,OUTPUT);

    analogWrite(PWMA, velocity_to_pwm(0.0f));
    analogWrite(PWMB, delta_to_pwm(0.0f));

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

    if (packetSize) {
        char incoming[128];
        int len = udp.read(incoming, sizeof(incoming) - 1);

        if (len > 0) {
            incoming[len] = '\0';
        }

        Serial.print("Recebido: ");
        Serial.println(incoming);

        float v, d;
        if (parseCommand(incoming, v, d)) {
            v_cmd = v;
            d_cmd = d;
        } else {
            Serial.println("Formato invalido");
        }
    }
    
    int pwm_v = velocity_to_pwm(v_cmd);
    int pwm_d = delta_to_pwm(d_cmd);

    if(v_cmd > 0){
        digitalWrite(STBY,HIGH);
        digitalWrite(BIN1,LOW);
        digitalWrite(BIN2,HIGH);
    }
    else if(v_cmd < 0){
        digitalWrite(STBY,HIGH);
        digitalWrite(BIN1,HIGH);
        digitalWrite(BIN2,LOW);
    }
    else{
        digitalWrite(STBY,LOW);
    }

    if(d_cmd > 0){
        digitalWrite(STBY,HIGH);
        digitalWrite(AIN1,LOW);
        digitalWrite(AIN2,HIGH);
    }
    else if(d_cmd < 0){
        digitalWrite(STBY,HIGH);
        digitalWrite(AIN1,HIGH);
        digitalWrite(AIN2,LOW);
    }
    else{
        digitalWrite(STBY,LOW);
    }
    
    analogWrite(PWMA,pwm_v);
    analogWrite(PWMB,pwm_d);
    //analogWrite(PWMA,255);
    //analogWrite(PWMB,255);
    
}
