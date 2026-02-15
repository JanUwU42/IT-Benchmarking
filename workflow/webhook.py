import argparse
import csv
import random
import time

import requests
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Webhook URLs for different providers
WEBHOOK_URLS = {
    "ollama": "http://localhost:5678/webhook/d15a6547-dd1d-4031-b6fa-fe92165249eb",
    "chatgpt": "http://localhost:5678/webhook/d15a6547-dd1d-4031-b6fa-fe92165249eb",
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark fake news detection using different LLM providers via webhooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webhook_trigger.py --provider ollama
  python webhook_trigger.py --provider chatgpt --samples 50
  python webhook_trigger.py -p ollama -s 200 --fake-csv data/fake.csv --true-csv data/true.csv
  python webhook_trigger.py --provider ollama --url http://custom-url.com/webhook
        """,
    )

    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=["ollama", "chatgpt"],
        required=True,
        help="LLM provider to use (ollama or chatgpt)",
    )

    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=100,
        help="Number of random samples to test (default: 100)",
    )

    parser.add_argument(
        "--fake-csv",
        type=str,
        default="fake.csv",
        help="Path to fake news CSV file (default: fake.csv)",
    )

    parser.add_argument(
        "--true-csv",
        type=str,
        default="true.csv",
        help="Path to true news CSV file (default: true.csv)",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom webhook URL (overrides provider default)",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )

    return parser.parse_args()


def load_csv(filepath, label):
    """Load CSV file and return list of dictionaries with source label."""
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["_source"] = label  # 'fake' or 'true'
            data.append(row)
    return data


def send_to_webhook(webhook_url, title, text):
    """Send title and text to the webhook and measure latency."""
    payload = {"title": title, "text": text}

    start_time = time.time()
    try:
        response = requests.post(webhook_url, json=payload)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        response.raise_for_status()
        return {"success": True, "response": response.json(), "latency_ms": latency}
    except requests.exceptions.JSONDecodeError:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "error": "Invalid JSON response",
            "latency_ms": latency,
            "raw_response": response.text if "response" in dir() else None,
        }
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {"success": False, "error": str(e), "latency_ms": latency}


def parse_prediction(response_data):
    """
    Parse the webhook response to extract prediction.
    Adjust this function based on your webhook's response format.
    """
    if not isinstance(response_data, dict):
        return None

    # Try common response field names
    for field in ["prediction", "label", "result", "classification", "output"]:
        if field in response_data:
            value = str(response_data[field]).lower().strip()
            if value in ["fake", "false", "0"]:
                return "fake"
            elif value in ["true", "real", "1"]:
                return "true"

    return None


def main():
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Determine webhook URL
    webhook_url = args.url if args.url else WEBHOOK_URLS[args.provider]

    print(f"{'=' * 60}")
    print(f"FAKE NEWS DETECTION BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Provider:    {args.provider.upper()}")
    print(f"Webhook URL: {webhook_url}")
    print(f"Samples:     {args.samples}")
    print(f"Fake CSV:    {args.fake_csv}")
    print(f"True CSV:    {args.true_csv}")
    if args.seed:
        print(f"Random Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    # Load both CSV files with labels
    try:
        fake_data = load_csv(args.fake_csv, "fake")
        true_data = load_csv(args.true_csv, "true")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Combine both datasets
    combined_data = fake_data + true_data

    # Take random samples
    sample_size = min(args.samples, len(combined_data))
    samples = random.sample(combined_data, sample_size)

    print(f"Sending {sample_size} samples to webhook...\n")

    # Tracking metrics
    y_true = []
    y_pred = []
    latencies = []
    invalid_responses = 0
    successful_requests = 0
    failed_requests = 0

    for i, sample in enumerate(samples, 1):
        title = sample.get("title", "")
        text = sample.get("text", "")
        actual_label = sample["_source"]

        result = send_to_webhook(webhook_url, title, text)
        latencies.append(result["latency_ms"])

        if result["success"]:
            successful_requests += 1
            prediction = parse_prediction(result["response"])

            if prediction is None:
                invalid_responses += 1
                print(
                    f"[{i}/{sample_size}] Invalid response format: {result['response']}"
                )
            else:
                y_true.append(actual_label)
                y_pred.append(prediction)
                is_correct = prediction == actual_label
                status = "✓" if is_correct else "✗"
                print(
                    f"[{i}/{sample_size}] {status} Actual: {actual_label}, Predicted: {prediction}, Latency: {result['latency_ms']:.2f}ms"
                )
        else:
            failed_requests += 1
            invalid_responses += 1
            print(
                f"[{i}/{sample_size}] Request failed: {result.get('error', 'Unknown error')}"
            )

    # Calculate and display metrics
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {args.provider.upper()}")
    print(f"{'=' * 60}")

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

    # Classification metrics
    if y_true and y_pred:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) * 100

        f1 = f1_score(y_true, y_pred, pos_label="fake", average="binary")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Accuracy:            {accuracy:.2f}%")
        print(f"   F1 Score (fake):     {f1:.4f}")
        print(f"   F1 Score (weighted): {f1_weighted:.4f}")
        print(f"   F1 Score (macro):    {f1_macro:.4f}")

        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=["fake", "true"]))

        cm = confusion_matrix(y_true, y_pred, labels=["fake", "true"])
        print(f"📉 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                 fake    true")
        print(f"   Actual fake   {cm[0][0]:<7} {cm[0][1]}")
        print(f"   Actual true   {cm[1][0]:<7} {cm[1][1]}")
    else:
        print("\n⚠️  No valid predictions to calculate classification metrics.")


if __name__ == "__main__":
    main()
