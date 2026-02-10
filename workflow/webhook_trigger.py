import csv
import random
import requests
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix

WEBHOOK_URL = "http://localhost:5678/webhook/d15a6547-dd1d-4031-b6fa-fe92165249eb"

def load_csv(filepath, label):
    """Load CSV file and return list of dictionaries with source label."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['_source'] = label  # 'fake' or 'true'
            data.append(row)
    return data

def send_to_webhook(title, text):
    """Send title and text to the webhook and measure latency."""
    payload = {
        "title": title,
        "text": text
    }

    start_time = time.time()
    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        response.raise_for_status()
        return {
            "success": True,
            "response": response.json(),
            "latency_ms": latency
        }
    except requests.exceptions.JSONDecodeError:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "error": "Invalid JSON response",
            "latency_ms": latency,
            "raw_response": response.text if 'response' in dir() else None
        }
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "latency_ms": latency
        }

def parse_prediction(response_data):
    """
    Parse the webhook response to extract prediction.
    Adjust this function based on your webhook's response format.
    Expected format: {"prediction": "fake"} or {"prediction": "true"}
    or {"label": "fake"/"true"} or {"result": "fake"/"true"}
    """
    if not isinstance(response_data, dict):
        return None

    # Try common response field names
    for field in ['prediction', 'label', 'result', 'classification', 'output']:
        if field in response_data:
            value = str(response_data[field]).lower().strip()
            if value in ['fake', 'false', '0']:
                return 'fake'
            elif value in ['true', 'real', '1']:
                return 'true'

    return None

def main():
    # Load both CSV files with labels
    fake_data = load_csv('fake.csv', 'fake')
    true_data = load_csv('true.csv', 'true')

    # Combine both datasets
    combined_data = fake_data + true_data

    # Take 100 random samples (or all if less than 100)
    sample_size = min(50, len(combined_data))
    samples = random.sample(combined_data, sample_size)

    print(f"Sending {sample_size} samples to webhook...\n")

    # Tracking metrics
    y_true = []  # Actual labels
    y_pred = []  # Predicted labels
    latencies = []
    invalid_responses = 0
    successful_requests = 0
    failed_requests = 0

    for i, sample in enumerate(samples, 1):
        title = sample.get('title', '')
        text = sample.get('text', '')
        actual_label = sample['_source']

        result = send_to_webhook(title, text)
        latencies.append(result['latency_ms'])

        if result['success']:
            successful_requests += 1
            prediction = parse_prediction(result['response'])

            if prediction is None:
                invalid_responses += 1
                print(f"[{i}/{sample_size}] Invalid response format: {result['response']}")
            else:
                y_true.append(actual_label)
                y_pred.append(prediction)
                is_correct = prediction == actual_label
                status = "✓" if is_correct else "✗"
                print(f"[{i}/{sample_size}] {status} Actual: {actual_label}, Predicted: {prediction}, Latency: {result['latency_ms']:.2f}ms")
        else:
            failed_requests += 1
            invalid_responses += 1
            print(f"[{i}/{sample_size}] Request failed: {result.get('error', 'Unknown error')}")

    # Calculate and display metrics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Latency statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\n📊 LATENCY METRICS:")
        print(f"   Average Latency: {avg_latency:.2f} ms")
        print(f"   Min Latency:     {min_latency:.2f} ms")
        print(f"   Max Latency:     {max_latency:.2f} ms")
        print(f"   Total Time:      {sum(latencies):.2f} ms")

    # Request statistics
    print(f"\n📡 REQUEST STATISTICS:")
    print(f"   Total Requests:      {sample_size}")
    print(f"   Successful:          {successful_requests}")
    print(f"   Failed:              {failed_requests}")
    print(f"   Invalid Responses:   {invalid_responses}")

    # Classification metrics (only if we have valid predictions)
    if y_true and y_pred:
        # Calculate accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) * 100

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, pos_label='fake', average='binary')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Accuracy:            {accuracy:.2f}%")
        print(f"   F1 Score (fake):     {f1:.4f}")
        print(f"   F1 Score (weighted): {f1_weighted:.4f}")
        print(f"   F1 Score (macro):    {f1_macro:.4f}")

        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=['fake', 'true']))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['fake', 'true'])
        print(f"📉 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                 fake    true")
        print(f"   Actual fake   {cm[0][0]:<7} {cm[0][1]}")
        print(f"   Actual true   {cm[1][0]:<7} {cm[1][1]}")
    else:
        print("\n⚠️  No valid predictions to calculate classification metrics.")

if __name__ == "__main__":
    main()
