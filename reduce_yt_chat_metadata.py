import json
import argparse
import os
import csv

def clean_youtube_chat(input_path, output_path=None, output_format='json'):
    # Determine default output filename
    if not output_path:
        base_name = os.path.splitext(input_path)[0]
        ext = 'tsv' if output_format == 'csv' else 'json'
        output_path = f"{base_name}_reduced.{ext}"

    optimized_data = []

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                actions = data.get("replayChatItemAction", {}).get("actions", [])
                
                for action in actions:
                    item = action.get("addChatItemAction", {}).get("item", {})
                    
                    if "liveChatTextMessageRenderer" in item:
                        renderer = item["liveChatTextMessageRenderer"]
                        
                        # Extract the Archive-Relative Timestamp
                        # This matches the time on the video player (e.g., "1:20:27")
                        ts_text = renderer.get("timestampText", {}).get("simpleText", "")
                        
                        # FILTER: Skip chats from before the stream started (negative timestamps)
                        if not ts_text or ts_text.startswith("-"):
                            continue
                        
                        # Extract Author
                        author = renderer.get("authorName", {}).get("simpleText", "Unknown")
                        
                        # Consolidate Message (Text + Emotes)
                        message_parts = []
                        runs = renderer.get("message", {}).get("runs", [])
                        for run in runs:
                            if "text" in run:
                                message_parts.append(run["text"])
                            elif "emoji" in run:
                                emoji_label = run["emoji"].get("shortcuts", [""])[0]
                                message_parts.append(emoji_label)
                        
                        full_message = " ".join(message_parts).strip()
                        
                        optimized_data.append({
                            "video_time": ts_text,
                            "author": author,
                            "message": full_message
                        })
                        
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    # Save to file
    if output_format == 'csv':
        keys = ["video_time", "author", "message"]
        with open(output_path, 'w', encoding='utf-8', newline='') as out_f:
            dict_writer = csv.DictWriter(out_f, fieldnames=keys, delimiter='\t')
            dict_writer.writeheader()
            dict_writer.writerows(optimized_data)
    else:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(optimized_data, out_f, indent=4, ensure_ascii=False)
    
    print(f"Cleanup complete! Kept {len(optimized_data)} on-stream messages.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean YouTube chat and align with video archive.")
    parser.add_argument("input", help="Path to the original .live_chat.json file")
    parser.add_argument("-o", "--output", help="Custom output path", default=None)
    parser.add_argument("--csv", action="store_true", help="Output as a tab-separated TSV")
    
    args = parser.parse_args()
    fmt = 'csv' if args.csv else 'json'
    clean_youtube_chat(args.input, args.output, output_format=fmt)
