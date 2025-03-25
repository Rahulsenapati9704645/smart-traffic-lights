// Include required libraries
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// Define the ESP8266 web server on port 80
ESP8266WebServer server(80);

// Define LED pins (D0 to D7)
const int ledPins[] = {D0, D1, D2, D3, D4, D5, D6, D7};
const int numLeds = 8;

void setup() {
    Serial.begin(115200);
    
    // Initialize LED pins as OUTPUT
    for (int i = 0; i < numLeds; i++) {
        pinMode(ledPins[i], OUTPUT);
        digitalWrite(ledPins[i], LOW); // Ensure all LEDs are off initially
    }

    // Connect to WiFi
    WiFi.begin("Airtel_Pydi raju", "Pydiraju@1460");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");
    Serial.println(WiFi.localIP());

    // Define the web server routes
    server.on("/message", HTTP_GET, []() {
        if (server.hasArg("text")) {
            String command = server.arg("text");
            Serial.println("Received command: " + command);
            
            // Control LEDs based on the received command
            if (command == "1_ON") {
                digitalWrite(D0, HIGH);
                digitalWrite(D3, HIGH);
                digitalWrite(D5, HIGH);
                digitalWrite(D7, HIGH);
                digitalWrite(D1, LOW);
                digitalWrite(D2, LOW);
                digitalWrite(D4, LOW);
                digitalWrite(D6, LOW);
            } else if (command == "2_ON") {
                digitalWrite(D1, HIGH);
                digitalWrite(D3, HIGH);
                digitalWrite(D4, HIGH);
                digitalWrite(D7, HIGH);
                digitalWrite(D0, LOW);
                digitalWrite(D2, LOW);
                digitalWrite(D5, LOW);
                digitalWrite(D6, LOW);
            } else if (command == "3_ON") {
                digitalWrite(D1, HIGH);
                digitalWrite(D3, HIGH);
                digitalWrite(D5, HIGH);
                digitalWrite(D6, HIGH);
                digitalWrite(D0, LOW);
                digitalWrite(D2, LOW);
                digitalWrite(D4, LOW);
                digitalWrite(D7, LOW);
            } else if (command == "4_ON") {
                digitalWrite(D1, HIGH);
                digitalWrite(D2, HIGH);
                digitalWrite(D5, HIGH);
                digitalWrite(D7, HIGH);
                digitalWrite(D0, LOW);
                digitalWrite(D3, LOW);
                digitalWrite(D4, LOW);
                digitalWrite(D6, LOW);
            } else {
                server.send(400, "text/plain", "Invalid Command");
                return;
            }
            server.send(200, "text/plain", "Command Executed");
        } else {
            server.send(400, "text/plain", "No Command Received");
        }
    });
    
    // Start the server
    server.begin();
}

void loop() {
    server.handleClient();
}
