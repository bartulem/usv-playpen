/* 
generateSyncPulses.ino

This arduino sketch generates a train of digital output pulses with random inter-pulse intervals.
This pulse train can be sent to any number of recording devices to allow precise synchronization of their
respective system clocks, even if data is lost.

By default, the Arduino will use the same random seed when booted up. Therefore if the random seed is not
initialized independently, we would get exactly the same pulse train every time. If the arduino were 
restarted during a recording, the re-occurrence of the same pulse sequence could be a big problem for 
correctly aligning the pulse trains. To avoid this possibility, this sketch attempts to randomly initialize
the seed of the random number generator, by using the intrinsic jitter in the Arduino's "watchdog" timer to 
obtain the initial seed value.

Tested on the Arduino Uno.

©bartulem 2024-03-14

*/

// Random seed includes
#include <avr/interrupt.h>
#include <avr/wdt.h>
#include <util/atomic.h>
#define randomSeed(s) srandom(s)
volatile uint32_t seed;  // These two variables can be reused in your program after the
volatile int8_t nrot;    // function CreateTrulyRandomSeed()executes in the setup() function

/*
Define arduino output pins. Change these values as necessary.
The LED_BUILTIN pin is helpful as an indicator that the program is running.
*/

#define OUTPUT_PINS 2
#define N_PINS 3
const int pins[N_PINS] = {2, 3, 4};
const int op[OUTPUT_PINS] = {7, 11};

// Define some parameters for the pulse train
#define INTERVAL_MIN 250    // Minimum IPI in ms
#define INTERVAL_MAX 1500   // Maximum IPI in ms
#define PULSE_LEN 250       // Pulse duration

void CreateTrulyRandomSeed()
{
  seed = 0;
  nrot = 32; // Must be at least 4, but more increased the uniformity of the produced 
             // seeds entropy.
  
  // The following five lines of code turn on the watch dog timer interrupt to create
  // the seed value
  cli();
  MCUSR = 0;
  _WD_CONTROL_REG |= (1<<_WD_CHANGE_BIT) | (1<<WDE);
  _WD_CONTROL_REG = (1<<WDIE);
  sei();
 
  while (nrot > 0);  // wait here until seed is created
 
  // The following five lines turn off the watch dog timer interrupt
  cli();                                             
  MCUSR = 0;                                         
  _WD_CONTROL_REG |= (1<<_WD_CHANGE_BIT) | (0<<WDE); 
  _WD_CONTROL_REG = (0<< WDIE);                      
  sei();                                             
}

ISR(WDT_vect)
{
  nrot--;
  seed = seed << 8;
  seed = seed ^ TCNT1L;
}

void setup() {
  
  // Initialize random seed
  Serial.begin(9600);
  CreateTrulyRandomSeed();
  randomSeed(seed);
  // Serial.print("Random seed = ");
  // Serial.print(seed);
  // Serial.print("\n");

  for(int j=0; j<OUTPUT_PINS; j++) {
    pinMode(op[j], OUTPUT);
    digitalWrite(op[j], LOW);
  }

  for(int i=0; i<N_PINS; i++) {
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], LOW);
  }
  
}

void loop() {
  
  // Generate the pulse
  for (int j=0; j<OUTPUT_PINS; j++){digitalWrite(op[j], HIGH);}
  for (int i=0; i<N_PINS; i++){digitalWrite(pins[i], HIGH);}
  
  delay(PULSE_LEN);
  
  for (int j=0; j<OUTPUT_PINS; j++){digitalWrite(op[j], LOW);}
  for (int i=0; i<N_PINS; i++){digitalWrite(pins[i], LOW);}
  
  // Wait for random inter-pulse interval
  int random_delay = random(INTERVAL_MIN, INTERVAL_MAX);
  Serial.println(random_delay);
  delay(random_delay);
  
}
