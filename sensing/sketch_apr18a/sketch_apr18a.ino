#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include "EMGFilters.h"

#define TIMING_DEBUG 1

#define SensorInputPin A1 // input pin number

EMGFilters myFilter;
int sampleRate = SAMPLE_FREQ_500HZ;
int humFreq = NOTCH_FREQ_60HZ;

static int Threshold = 0;
static int Bias = 200;
unsigned long timeStamp;
unsigned long timeBudget;
int maxValue = 0;

void setup() {
    myFilter.init(sampleRate, humFreq, true, true, true);
    Serial.begin(115200);
    timeBudget = 1e6 / sampleRate;
}

int history = 0;
int histime = 0;

void loop() {
    timeStamp = micros();
    int Value = analogRead(SensorInputPin);
    int grad = (history - Value)/(histime-timeStamp);
    if (TIMING_DEBUG) {
        Serial.println(abs(Value));
    }
    history = Value;
    histime = micros();
    delayMicroseconds(500);
}
