import time
import json
import requests
import statistics
import paho.mqtt.client as mqtt

# Configuration
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "parking/benchmark"

HTTP_URL = "http://127.0.0.1:8081/api/iot/detection"

# Data structures to store measurements
mqtt_latencies = []
received_count = 0
total_messages = 50

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Connection to MQTT broker failed with code {rc}")

def on_message(client, userdata, msg):
    global received_count
    try:
        payload = json.loads(msg.payload.decode())
        sent_time = payload.get("timestamp")
        if sent_time:
            # Calculate roundtrip latency in milliseconds
            latency = (time.perf_counter() - sent_time) * 1000
            mqtt_latencies.append(latency)
            received_count += 1
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")

def run_mqtt_benchmark():
    global received_count, mqtt_latencies
    mqtt_latencies = []
    received_count = 0
    
    print("\n--- Starting MQTT Latency Benchmark ---")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Could not connect to MQTT Broker on {MQTT_BROKER}:{MQTT_PORT}: {e}")
        return None
        
    client.loop_start()
    
    # Wait a moment for subscription to complete
    time.sleep(0.5)
    
    print(f"Sending {total_messages} benchmark messages via MQTT...")
    for i in range(total_messages):
        payload = {
            "msg_id": i,
            "timestamp": time.perf_counter()
        }
        client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        time.sleep(0.02)  # 20ms gap between messages
        
    # Wait for all messages to be received
    timeout = 3.0
    start_wait = time.time()
    while received_count < total_messages and (time.time() - start_wait) < timeout:
        time.sleep(0.1)
        
    client.loop_stop()
    client.disconnect()
    
    if not mqtt_latencies:
        print("No MQTT messages were successfully received.")
        return None
        
    return {
        "sent": total_messages,
        "received": received_count,
        "min": min(mqtt_latencies),
        "max": max(mqtt_latencies),
        "avg": statistics.mean(mqtt_latencies),
        "std_dev": statistics.stdev(mqtt_latencies) if len(mqtt_latencies) > 1 else 0.0
    }

def run_http_benchmark():
    print("\n--- Starting HTTP Response Time Benchmark ---")
    http_latencies = []
    
    payload = {
        "parkingId": 1,
        "ipRaspberry": "127.0.0.1",
        "idCamera": "CAM-BENCHMARK",
        "detections": [
            {"placeId": 1, "etat": 0, "confidence": 0.95}
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending {total_messages} POST requests to {HTTP_URL}...")
    success_count = 0
    
    for i in range(total_messages):
        start_time = time.perf_counter()
        try:
            # We use a short timeout to prevent blocking
            response = requests.post(HTTP_URL, json=payload, headers=headers, timeout=2.0)
            latency = (time.perf_counter() - start_time) * 1000
            http_latencies.append(latency)
            if response.status_code in [200, 201]:
                success_count += 1
        except Exception as e:
            # Even if it errors out (e.g. 500 or timeout), we record the time taken
            latency = (time.perf_counter() - start_time) * 1000
            http_latencies.append(latency)
            
        time.sleep(0.02)  # 20ms gap between requests
        
    if not http_latencies:
        print("No HTTP requests completed.")
        return None
        
    return {
        "sent": total_messages,
        "success": success_count,
        "min": min(http_latencies),
        "max": max(http_latencies),
        "avg": statistics.mean(http_latencies),
        "std_dev": statistics.stdev(http_latencies) if len(http_latencies) > 1 else 0.0
    }

if __name__ == "__main__":
    mqtt_results = run_mqtt_benchmark()
    http_results = run_http_benchmark()
    
    print("\n" + "=" * 60)
    print("                 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    if mqtt_results:
        print("MQTT Protocol (Pub/Sub Transit):")
        print(f"  - Messages Sent/Received: {mqtt_results['sent']}/{mqtt_results['received']}")
        print(f"  - Average Latency:        {mqtt_results['avg']:.2f} ms")
        print(f"  - Min/Max Latency:        {mqtt_results['min']:.2f} ms / {mqtt_results['max']:.2f} ms")
        print(f"  - Jitter (Std Dev):       {mqtt_results['std_dev']:.2f} ms")
    else:
        print("MQTT Protocol: Benchmark failed (is the broker running on port 1883?)")
        
    print("-" * 60)
    
    if http_results:
        print("HTTP REST Protocol (Request/Response API):")
        print(f"  - Requests Sent (Success): {http_results['sent']} ({http_results['success']})")
        print(f"  - Average Latency:         {http_results['avg']:.2f} ms")
        print(f"  - Min/Max Latency:         {http_results['min']:.2f} ms / {http_results['max']:.2f} ms")
        print(f"  - Jitter (Std Dev):        {http_results['std_dev']:.2f} ms")
    else:
        print("HTTP REST Protocol: Benchmark failed (is the Spring Boot app running on port 8081?)")
        
    print("=" * 60)
    
    if mqtt_results and http_results:
        speedup = http_results['avg'] / mqtt_results['avg']
        print(f"MQTT is approx {speedup:.1f}x FASTER than HTTP REST in this local test environment.")
        print("=" * 60)
