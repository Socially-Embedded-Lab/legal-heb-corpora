#!/usr/bin/env python3
"""
Extract Indictment Facts from Legal Verdicts

- Uses GPT-4 + heuristics to identify relevant sections
- Identifies start/end of factual section
- Sends to GPT for clean extraction
"""

import pandas as pd
import os
import re
from openai import OpenAI
import gc
from tqdm import tqdm
import time
import argparse

# ========== CONFIGURATION ==========
# Set your OpenAI API key as environment variable: export OPENAI_API_KEY="your-key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Pattern Definitions ==========
START_PARTS = ["עובדותם", "כללי", "כתב האישום", "האישום", "אישום", "רקע", "גזר", "דין", "פסק","מבוא","הרשעת" ,"בעניינו","עבירות","הורשע","עובדות","השתלשלות", "ג ז ר",  "ד י ן","פתח דבר","פתח"]
END_PARTS = ["טענות", "עמדת", "תסקיר","תסקירי", "שירות", "מבחן", "דיון", "התסקיר","טיעוני", "הצדדים", "צדדים", "והכרעה",  "ראיות","החלטה"]
EXCLUDED_START_PARTS = ["כתבי עת", "חקיקה שאוזכרה", "חקיקה", "ציטוטים", "מקורות"]

# ========== Helper Functions ==========
def extract_indictment_facts(df):
    """
    מחלץ את עובדות כתב האישום מתוך DataFrame של פסק דין.
    מזהה את תחילת וסוף החלק העובדתי לפי שמות החלקים.
    """
    if df.empty or "part" not in df.columns or "text" not in df.columns:
        return "❌ No indictment facts found", None, None, 0

    df["part"] = df["part"].astype(str).str.strip()
    
    # Find start part
    start_row = df[df["part"].str.contains('|'.join(START_PARTS), case=False, na=False, regex=True)]
    if not start_row.empty:
        excluded_mask = start_row["part"].str.contains('|'.join(EXCLUDED_START_PARTS), case=False, na=False, regex=True)
        start_row = start_row[~excluded_mask]
    
    if start_row.empty:
        # Fallback: search for indictment keywords in text
        for idx, row in df.iterrows():
            text_content = str(row.get("text", "")).strip() if pd.notna(row.get("text")) else ""
            if text_content:
                text_lower = text_content.casefold()
                indictment_keywords = ["הורשע", "הרשענו", "מצאנו להרשיעו", "כתב אישום", "הנאשם הורשע"]
                if any(keyword in text_lower for keyword in indictment_keywords):
                    start_idx = idx
                    start_part_name = df.loc[idx, "part"]
                    normalized_start_part = re.sub(r"\s+", " ", str(start_part_name).strip().casefold())
                    has_start = True
                    break
        else:
            start_idx = 0
            start_part_name = "❌ No start found (use index 0)"
            normalized_start_part = None
            has_start = False
    else:
        start_idx = start_row.index.min()
        start_part_name = df.loc[start_idx, "part"]
        normalized_start_part = re.sub(r"\s+", " ", str(start_part_name).strip().casefold())
        has_start = True

    # Find end part
    end_mask = (
        (df.index > start_idx) &
        (df["part"].str.contains('|'.join(END_PARTS), case=False, na=False, regex=True))
    )
    end_row = df.loc[end_mask]
    end_candidates = end_row.index.tolist()
    valid_end_idx = None

    for candidate_idx in end_candidates:
        candidate_part = str(df.loc[candidate_idx, "part"]).strip()
        
        # Skip if matches START_PARTS
        if candidate_part:
            candidate_series = pd.Series([candidate_part])
            matches_start_pattern = candidate_series.str.contains('|'.join(START_PARTS), case=False, na=False, regex=True).iloc[0]
            if matches_start_pattern:
                continue
        
        # Skip if same as start part
        if has_start:
            normalized_candidate = re.sub(r"\s+", " ", candidate_part.casefold())
            if normalized_candidate == normalized_start_part:
                continue
            if normalized_start_part and normalized_start_part in normalized_candidate:
                continue
        
        valid_end_idx = candidate_idx
        break

    if valid_end_idx is not None:
        end_idx = valid_end_idx
        end_part_name = df.loc[end_idx, "part"]
    else:
        end_idx = len(df)
        end_part_name = "❌ No end found (used full text)"

    parts_count = end_idx - start_idx
    
    # Extract text grouped by part
    extracted_sections = []
    current_part = None
    current_text_parts = []
    
    for idx in range(start_idx, end_idx):
        row = df.loc[idx]
        part_name = str(row["part"]).strip()
        text_content = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        
        if not text_content:
            continue
            
        if part_name != current_part:
            if current_part is not None and current_text_parts:
                extracted_sections.append(f"{current_part}:")
                extracted_sections.append("\n".join(current_text_parts))
                extracted_sections.append("")
            
            current_part = part_name
            current_text_parts = [text_content]
        else:
            current_text_parts.append(text_content)
    
    if current_part is not None and current_text_parts:
        extracted_sections.append(f"{current_part}:")
        extracted_sections.append("\n".join(current_text_parts))
    
    extracted_text = "\n".join(extracted_sections)
    
    # Validate content
    if extracted_text:
        extracted_text_lower = extracted_text.casefold()
        indictment_keywords = [
            "הורשע", "הרשענו", "מצאנו להרשיעו", "כתב אישום", "כתב האישום",
            "על פי הודאתו", "על פי הודאת", "הודה", "הנאשם הורשע", "הנאשם הודה",
            "הסדר טיעון", "בעבירות", "לפי סעיף", "לפי סעיפים"
        ]
        
        has_indictment_content = any(keyword in extracted_text_lower for keyword in indictment_keywords)
        
        if not has_indictment_content:
            citation_indicators = ["חוק העונשין", "פקודת", "תקנות", "ע\"פ", "ע\"א"]
            has_citations = any(indicator in extracted_text_lower for indicator in citation_indicators)
            
            if has_citations and len(extracted_text.split()) < 50:
                return "❌ No indictment facts found", start_part_name, end_part_name, parts_count
    
    return extracted_text.strip() if extracted_text else "❌ No indictment facts found", start_part_name, end_part_name, parts_count


def extract_facts_with_gpt(text):
    """
    שולח את הטקסט המחולץ ל-GPT לחילוץ נקי של עובדות כתב האישום.
    """
    if text == "❌ No indictment facts found":
        return "GPT extraction error"

    prompt = f"""
תפקידך הוא לחלץ מידע משפטי מתוך טקסט של גזר דין.
המטרה שלך היא למצוא את "הסיפור העובדתי" - בגין מה הורשע הנאשם ומה בדיוק קרה שם.

עליך לחלץ שני חלקים:
1. **פסקת האישום/ההרשעה**: המשפט הפורמלי שקובע במה הנאשם הורשע (סעיפי חוק, סוג העבירה, הודאה/הכחשה).
2. **תיאור העובדות**: הסיפור המלא של המקרה (מה קרה, מתי, איפה, מי המעורבים).

הנחיות לביצוע:
1. חפש עוגנים כמו: "הנאשם הורשע", "על פי עובדות כתב האישום", "כתב האישום המתוקן".
2. אם הטקסט מכיל תיאור עובדתי מיד לאחר ההרשעה - העתק את כולו.
3. **אל תסכם**. העתק את הטקסט המקורי מילה במילה.
4. **מתי לעצור?** הפסק להעתיק כאשר הטקסט עובר לנושאים אחרים כגון: "תסקיר שירות המבחן", "טיעונים לעונש", "דיון והכרעה".

טקסט לעיבוד:
{text}

החזר את הפלט בפורמט הבא בלבד:
<פסקת כתב האישום>

<פסקת עובדות כתב האישום>
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview", 
        messages=[
            {"role": "system", "content": "אתה מודל בינה מלאכותית שתפקידו לחלץ עובדות מכתבי אישום בטקסטים משפטיים בעברית, מבלי לפרש, לסכם או לשנות את הנוסח המקורי."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


def is_valid_extraction(gpt_text):
    """בודק אם החילוץ תקין"""
    if pd.isna(gpt_text) or not isinstance(gpt_text, str):
        return False
    gpt_text = gpt_text.strip()
    if gpt_text == "" or gpt_text == "GPT extraction error":
        return False
    return True


def process_verdicts(csv_directory, output_file, target_csv_path=None, dry_run=False):
    """
    מעבד את כל פסקי הדין בתיקייה.
    """
    # Load target verdicts if specified
    unique_verdicts = None
    if target_csv_path and os.path.exists(target_csv_path):
        target_df = pd.read_csv(target_csv_path)
        unique_verdicts = set(target_df['verdict_1'].astype(str).str.strip().unique()) | \
                         set(target_df['verdict_2'].astype(str).str.strip().unique())
        print(f"✅ Processing only {len(unique_verdicts)} unique verdicts from target CSV")
    
    # Load existing data
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file)
    else:
        processed_df = pd.DataFrame(columns=["verdict", "extracted_facts", "extracted_gpt_facts", "start_part", "end_part", "parts_count"])

    processed_df["verdict"] = processed_df["verdict"].astype(str).str.strip()
    
    # Process files
    file_list = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]
    failed_verdicts = []

    for filename in tqdm(file_list, desc="Processing verdicts"):
        file_path = os.path.join(csv_directory, filename)
        try:
            df = pd.read_csv(file_path)
            verdict_id = str(df["verdict"].iloc[0]).strip()
            
            # Skip if not in target
            if unique_verdicts is not None and verdict_id not in unique_verdicts:
                continue

            # Extract facts
            extracted_facts, start_part, end_part, parts_count = extract_indictment_facts(df)
            
            if dry_run:
                continue

            # Check if already processed
            existing_rows = processed_df[processed_df["verdict"] == verdict_id]
            if not existing_rows.empty:
                existing_gpt = existing_rows["extracted_gpt_facts"].iloc[0]
                if is_valid_extraction(existing_gpt):
                    print(f"⏭️ Skipping GPT for verdict {verdict_id} (valid extraction exists)")
                    continue

            # Run GPT extraction
            extracted_gpt_facts = extract_facts_with_gpt(extracted_facts)
            
            # Save or update
            existing_rows = processed_df[processed_df["verdict"] == verdict_id]
            if not existing_rows.empty:
                idx = existing_rows.index[0]
                processed_df.at[idx, "extracted_facts"] = extracted_facts
                processed_df.at[idx, "extracted_gpt_facts"] = extracted_gpt_facts
                processed_df.at[idx, "start_part"] = start_part
                processed_df.at[idx, "end_part"] = end_part
                processed_df.at[idx, "parts_count"] = parts_count
            else:
                new_row = pd.DataFrame([{
                    "verdict": verdict_id,
                    "extracted_facts": extracted_facts,
                    "extracted_gpt_facts": extracted_gpt_facts,
                    "start_part": start_part,
                    "end_part": end_part,
                    "parts_count": parts_count
                }])
                processed_df = pd.concat([processed_df, new_row], ignore_index=True)
            
            processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            time.sleep(1)  # Rate limiting

        except Exception as e:
            failed_verdicts.append({"verdict": filename, "reason": str(e)})

        gc.collect()

    # Save failures
    if failed_verdicts:
        failed_file = output_file.replace(".csv", "_failed.csv")
        pd.DataFrame(failed_verdicts).to_csv(failed_file, index=False, encoding="utf-8-sig")
        print(f"⚠️ Saved {len(failed_verdicts)} failed verdicts to {failed_file}")

    return processed_df


def main():
    parser = argparse.ArgumentParser(description="חילוץ עובדות כתב האישום מפסקי דין")
    parser.add_argument("--domain", choices=["drugs", "weapon"], required=True,
                       help="דומיין לעיבוד: drugs או weapon")
    parser.add_argument("--base-path", required=True,
                       help="נתיב בסיס לתיקיית הדומיין")
    parser.add_argument("--dry-run", action="store_true",
                       help="הרצה יבשה (ללא קריאות API)")
    parser.add_argument("--process-all", action="store_true",
                       help="עיבוד כל הקבצים (לא רק מתוך target.csv)")
    
    args = parser.parse_args()
    
    csv_directory = os.path.join(args.base_path, "verdict_csv")
    output_dir = os.path.join(args.base_path, "gpt")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "processed_verdicts_with_gpt.csv")
    target_csv = None if args.process_all else os.path.join(args.base_path, "target.csv")
    
    print(f"🔄 Processing {args.domain} domain")
    print(f"   CSV directory: {csv_directory}")
    print(f"   Output file: {output_file}")
    if target_csv:
        print(f"   Target file: {target_csv}")
    
    process_verdicts(csv_directory, output_file, target_csv, args.dry_run)
    print("✅ Done!")


if __name__ == "__main__":
    main()



