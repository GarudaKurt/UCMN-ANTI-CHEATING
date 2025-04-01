const int LED_PIN = 13;

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW); // Ensure LED starts OFF
    Serial.begin(115200);  // Match Python's baud rate
}

void loop() {
    if (Serial.available()) {
        String command = Serial.readString();  // Read full input
        command.trim(); // Remove extra spaces/newlines

        if (command == "on") {
            digitalWrite(LED_PIN, HIGH);  // Turn LED ON
            Serial.println("LED is ON");
        } 
        else if (command == "off") {
            digitalWrite(LED_PIN, LOW);  // Turn LED OFF
            Serial.println("LED is OFF");
        } 
        else {
            Serial.println("Invalid command.");  // Debugging
        }
    }
}
