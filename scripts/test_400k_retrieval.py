import json
import time
import requests

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# 5 Target Needles placed at exact depth intervals
NEEDLES = {
    "depth_05": {"depth": 0.05, "key": "ALPHA_7701", "fact": "The secret access code for Vault Alpha is: 7701-XRAY."},
    "depth_25": {"depth": 0.25, "key": "BRAVO_9912", "fact": "The encryption key for Satellite Bravo is: 9912-TANGO."},
    "depth_50": {"depth": 0.50, "key": "CHARLIE_3345", "fact": "The primary password for Sector Charlie is: 3345-DELTA."},
    "depth_75": {"depth": 0.75, "key": "DELTA_8820", "fact": "The emergency override code for Reactor Delta is: 8820-SIERRA."},
    "depth_98": {"depth": 0.98, "key": "ECHO_1104", "fact": "The launch authorization code for Platform Echo is: 1104-ZULU."}
}

DISTRACTOR_PARAGRAPH = (
    "The operational telemetry indicates continuous stabilization of subsystem parameters across all physical channels. "
    "Sensors record nominal voltage fluctuations within the expected tolerances, while ambient temperature gradients remain "
    "well within standard engineering thresholds. Routine buffer flushes execute on schedule with zero unhandled exceptions, "
    "and cross-rack network routing tables maintain converged state across all primary and secondary gateways. "
)

def build_400k_payload(target_words=300000): # ~300k words ≈ 400k tokenizer tokens
    print(f"[1/3] Generating ~400,000-token distractor corpus with 5 embedded needles...")
    t0 = time.perf_counter()
    
    # Calculate word insertions
    needle_positions = {k: int(v["depth"] * target_words) for k, v in NEEDLES.items()}
    paragraph_words = len(DISTRACTOR_PARAGRAPH.split())
    
    corpus_parts = []
    current_word_count = 0
    inserted = set()
    
    while current_word_count < target_words:
        # Check if any needle needs insertion at this position
        for n_id, target_pos in needle_positions.items():
            if n_id not in inserted and current_word_count >= target_pos:
                corpus_parts.append(f"\n\n*** SPECIAL DIRECTIVE ***\n{NEEDLES[n_id]['fact']}\n*** END DIRECTIVE ***\n\n")
                current_word_count += len(NEEDLES[n_id]["fact"].split()) + 6
                inserted.add(n_id)
                print(f"  -> Inserted needle [{n_id}] at ~{target_pos:,} words ({NEEDLES[n_id]['depth']*100:.0f}% depth)")
        
        corpus_parts.append(DISTRACTOR_PARAGRAPH)
        current_word_count += paragraph_words

    full_context = "".join(corpus_parts)
    print(f"Generated {current_word_count:,} words in {time.perf_counter() - t0:.2f} s.\n")
    return full_context

def run_retrieval_query(context):
    query = (
        "Based strictly on the provided text, answer all 5 of the following questions clearly:\n"
        "1. What is the secret access code for Vault Alpha?\n"
        "2. What is the encryption key for Satellite Bravo?\n"
        "3. What is the primary password for Sector Charlie?\n"
        "4. What is the emergency override code for Reactor Delta?\n"
        "5. What is the launch authorization code for Platform Echo?"
    )

    messages = [
        {"role": "system", "content": "You are a precise information retrieval assistant. Answer directly using exact facts from the text."},
        {"role": "user", "content": f"DOCUMENT:\n{context}\n\nINSTRUCTION:\n{query}"}
    ]

    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False
    }

    print(f"[2/3] Sending request to {SERVER_URL} (this will take ~4-5 minutes for cold prefill)...")
    t0 = time.perf_counter()
    response = requests.post(SERVER_URL, json=payload, timeout=900)
    elapsed = time.perf_counter() - t0
    
    if response.status_code != 200:
        print(f"[ERROR] Request failed with HTTP {response.status_code}: {response.text}")
        return

    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    
    print(f"\n[3/3] Completed in {elapsed:.2f} s")
    print(f"  Prompt tokens:    {usage.get('prompt_tokens', 'n/a'):,}")
    print(f"  Generated tokens: {usage.get('completion_tokens', 'n/a'):,}")
    print("\n=== MODEL OUTPUT ===")
    print(reply)
    print("====================\n")
    
    # Audit recall
    print("=== NEEDLE VERIFICATION ===")
    passed = 0
    for n_id, n_data in NEEDLES.items():
        found = n_data["key"].split("_")[1] in reply or n_data["fact"] in reply
        status = "[PASSED]" if found else "[FAILED]"
        print(f"Needle {n_id} ({n_data['depth']*100:4.1f}% depth): {status}")
        if found: passed += 1
    
    print(f"\nFinal Score: {passed}/5 Needles Retrieved ({passed/5*100:.1f}%)")

if __name__ == "__main__":
    context = build_400k_payload()
    run_retrieval_query(context)