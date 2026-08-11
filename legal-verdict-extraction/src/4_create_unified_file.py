#!/usr/bin/env python3
"""
Create Unified Output File for Supervisors

Combines all extracted data into unified CSV files:
- verdict: Case identifier
- indictment_facts: Extracted conviction facts
- citations: Cited cases for sentencing policy [case_id, citation_paragraph]
- sentencing_range: Punishment range declared by judge

Domains: drugs, weapon, 5k (mixed)
"""

import pandas as pd
import os
from pathlib import Path
import json
from collections import defaultdict

# ========== PATHS ==========
BASE_DIR = Path(__file__).parent.parent  # innovation_submission/
SCRIPTS_DIR = Path(__file__).parent  # scripts/
NEW_TRY_DIR = BASE_DIR.parent  # new_try/
DRUGS_DIR = NEW_TRY_DIR / "drugs"
WEAPON_DIR = NEW_TRY_DIR / "weapon"
FIVE_K_DIR = NEW_TRY_DIR.parent / "innovation_task" / "5k"  # 5k dataset
PRE_PROCESS_DIR = NEW_TRY_DIR.parent / "pre-process"
OUTPUT_DIR = BASE_DIR / "results"  # repo-root/results/

# ========== DATA SOURCES ==========
# Indictment facts
DRUGS_INDICTMENT_FACTS = DRUGS_DIR / "gpt" / "processed_verdicts_with_gpt.csv"
WEAPON_INDICTMENT_FACTS = WEAPON_DIR / "gpt" / "processed_verdicts_with_gpt.csv"
FIVE_K_INDICTMENT_FACTS = FIVE_K_DIR / "gpt" / "processed_verdicts_with_gpt.csv"

# Citations
DRUGS_CITATIONS_DIR = DRUGS_DIR / "verdicts_tagged_citations"
WEAPON_CITATIONS_DIR = WEAPON_DIR / "verdicts_tagged_citations"
FIVE_K_CITATIONS_DIR = FIVE_K_DIR / "gpt" / "verdict_tagged_citations"

# Sentencing range (from pre-process folder and innovation_task)
SENTENCING_RANGE_DRUGS = PRE_PROCESS_DIR / "gpt_fewshot_classifier_extractor_results_verdict_csv.csv"
SENTENCING_RANGE_WEAPON = PRE_PROCESS_DIR / "gpt_fewshot_classifier_extractor_results_weapon.csv"  # Main weapon file (1394 POSITIVE)
SENTENCING_RANGE_5K = SCRIPTS_DIR / "gpt_fewshot_classifier_extractor_results_5k_verdict_csv.csv"  # Just extracted!
SENTENCING_RANGE_WEAPON_GT = NEW_TRY_DIR.parent / "innovation_task" / "features_gt_weapom - מתחם ענישה - שופט+ נרמול.csv"
SENTENCING_RANGE_INNOVATION = NEW_TRY_DIR.parent / "innovation_task" / "gpt_fewshot_classifier_extractor_results.csv"

# Target files (for filtering)
DRUGS_TARGET = DRUGS_DIR / "target.csv"
WEAPON_TARGET = WEAPON_DIR / "target.csv"


def load_indictment_facts(path: Path) -> pd.DataFrame:
    """Load indictment facts from GPT processed verdicts."""
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    # Keep only relevant columns
    return df[['verdict', 'extracted_facts', 'extracted_gpt_facts']].copy()


def load_citations(citations_dir: Path) -> dict:
    """
    Load all citations from individual verdict files.
    Returns: {verdict_id: [(cited_case, citation_text, predicted_label), ...]}
    """
    citations_dict = defaultdict(list)
    
    if not citations_dir.exists():
        print(f"⚠️ Citations directory not found: {citations_dir}")
        return citations_dict
    
    for csv_file in citations_dir.glob("*.csv"):
        verdict_id = csv_file.stem  # filename without extension
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                citation_entry = {
                    'cited_case': row.get('citation', ''),
                    'citation_text': row.get('extracted_text', ''),
                    'context': row.get('context_text', ''),
                    'predicted_label': row.get('predicted_label', '')
                }
                citations_dict[verdict_id].append(citation_entry)
        except Exception as e:
            print(f"⚠️ Error reading {csv_file}: {e}")
    
    return citations_dict


def load_sentencing_range(path: Path, is_weapon_gt: bool = False) -> pd.DataFrame:
    """Load sentencing range data."""
    if not path.exists():
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path, low_memory=False)
    
    # Handle weapon GT file with different column names
    if is_weapon_gt:
        df = df.rename(columns={
            'case': 'verdict',
            'מתחם ענישה - שופט': 'range_str',
            'low-month': 'range_low',
            'high-month': 'range_high'
        })
        df['classification'] = 'POSITIVE'  # GT data is positive
        df['confidence'] = 'גבוהה (GT)'
        df['sentence'] = df['range_str']
    
    # Keep only relevant columns
    cols = ['verdict', 'classification', 'sentence', 'confidence', 
            'range_low', 'range_high', 'range_str']
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].copy()


def get_target_verdicts(target_file: Path) -> set:
    """Get all unique verdict IDs from target file."""
    if not target_file.exists():
        print(f"⚠️ Target file not found: {target_file}")
        return set()
    
    df = pd.read_csv(target_file)
    verdicts = set(df['verdict_1'].unique()) | set(df['verdict_2'].unique())
    return verdicts


def create_unified_file(domain: str, 
                        indictment_facts_df: pd.DataFrame,
                        citations_dict: dict,
                        sentencing_df: pd.DataFrame,
                        target_verdicts: set,
                        output_path: Path) -> pd.DataFrame:
    """Create unified file for a domain."""
    
    print(f"\n📊 Creating unified file for {domain}...")
    
    # Get all verdicts from indictment facts
    all_verdicts = set(indictment_facts_df['verdict'].unique())
    
    # NOTE: Don't filter by target - include ALL verdicts for unified file
    # if target_verdicts:
    #     all_verdicts = all_verdicts & target_verdicts
    #     print(f"   📌 Filtered to {len(all_verdicts)} target verdicts")
    print(f"   📌 Using all {len(all_verdicts)} verdicts (no target filtering)")
    
    rows = []
    for verdict in all_verdicts:
        row = {'verdict': verdict, 'domain': domain}
        
        # Add indictment facts
        facts_row = indictment_facts_df[indictment_facts_df['verdict'] == verdict]
        if not facts_row.empty:
            row['indictment_facts'] = facts_row.iloc[0].get('extracted_gpt_facts', '')
            row['indictment_facts_raw'] = facts_row.iloc[0].get('extracted_facts', '')
        else:
            row['indictment_facts'] = ''
            row['indictment_facts_raw'] = ''
        
        # Add citations (as JSON string for easy reading)
        verdict_citations = citations_dict.get(verdict, [])
        if verdict_citations:
            # Format: list of [cited_case, citation_text]
            citations_formatted = [
                {'cited_case': c['cited_case'], 'citation_text': c['citation_text']}
                for c in verdict_citations
            ]
            row['citations_json'] = json.dumps(citations_formatted, ensure_ascii=False)
            row['citations_count'] = len(verdict_citations)
        else:
            row['citations_json'] = '[]'
            row['citations_count'] = 0
        
        # Add sentencing range
        sent_row = sentencing_df[sentencing_df['verdict'] == verdict]
        if not sent_row.empty:
            row['sentencing_classification'] = sent_row.iloc[0].get('classification', '')
            row['sentencing_sentence'] = sent_row.iloc[0].get('sentence', '')
            row['sentencing_confidence'] = sent_row.iloc[0].get('confidence', '')
            row['sentencing_range_low'] = sent_row.iloc[0].get('range_low', '')
            row['sentencing_range_high'] = sent_row.iloc[0].get('range_high', '')
            row['sentencing_range_str'] = sent_row.iloc[0].get('range_str', '')
        else:
            row['sentencing_classification'] = ''
            row['sentencing_sentence'] = ''
            row['sentencing_confidence'] = ''
            row['sentencing_range_low'] = ''
            row['sentencing_range_high'] = ''
            row['sentencing_range_str'] = ''
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"   ✅ Saved {len(result_df)} verdicts to {output_path}")
    
    return result_df


def create_citations_for_tagging(citations_dict: dict, domain: str, output_path: Path, sample_size: int = 200):
    """Create a sample of citations for manual tagging."""
    
    print(f"\n📋 Creating citations sample for {domain} tagging...")
    
    all_citations = []
    for verdict, citations in citations_dict.items():
        for c in citations:
            all_citations.append({
                'verdict': verdict,
                'cited_case': c['cited_case'],
                'citation_text': c['citation_text'],
                'context': c['context'],
                'predicted_label': c['predicted_label'],
                'manual_label': '',  # Empty for human tagging
                'notes': ''  # For annotator notes
            })
    
    if len(all_citations) == 0:
        print(f"   ⚠️ No citations found for {domain}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_citations)
    
    # Sample if needed
    if len(df) > sample_size:
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"   📌 Sampled {sample_size} from {len(df)} total citations")
    else:
        sampled_df = df
        print(f"   📌 Using all {len(df)} citations (less than {sample_size})")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved to {output_path}")
    
    return sampled_df


def create_indictment_facts_for_llm_judge(indictment_df: pd.DataFrame, domain: str, output_path: Path, sample_size: int = 100):
    """Create sample of indictment facts for LLM as judge evaluation."""
    
    print(f"\n📋 Creating indictment facts sample for {domain} LLM evaluation...")
    
    df = indictment_df.copy()
    df['domain'] = domain
    df['llm_quality_score'] = ''  # For LLM judge
    df['llm_reasoning'] = ''  # For LLM judge reasoning
    
    # Sample if needed
    if len(df) > sample_size:
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"   📌 Sampled {sample_size} from {len(df)} total verdicts")
    else:
        sampled_df = df
        print(f"   📌 Using all {len(df)} verdicts (less than {sample_size})")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved to {output_path}")
    
    return sampled_df


def create_sentencing_for_tagging(sentencing_df: pd.DataFrame, domain: str, output_path: Path, sample_size: int = 100):
    """Create sample of sentencing range for manual verification."""
    
    print(f"\n📋 Creating sentencing range sample for {domain} tagging...")
    
    df = sentencing_df.copy()
    df['domain'] = domain
    df['manual_verification'] = ''  # For human verification
    df['corrected_range'] = ''  # If correction needed
    
    # Sample if needed
    if len(df) > sample_size:
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"   📌 Sampled {sample_size} from {len(df)} total verdicts")
    else:
        sampled_df = df
        print(f"   📌 Using all {len(df)} verdicts (less than {sample_size})")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved to {output_path}")
    
    return sampled_df


def main():
    print("=" * 70)
    print("יצירת קובץ מאוחד למנחים")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== DRUGS DOMAIN ==========
    print("\n" + "=" * 40)
    print("🧪 DRUGS DOMAIN")
    print("=" * 40)
    
    # Load data
    drugs_indictment = load_indictment_facts(DRUGS_INDICTMENT_FACTS)
    drugs_citations = load_citations(DRUGS_CITATIONS_DIR)
    drugs_sentencing = load_sentencing_range(SENTENCING_RANGE_DRUGS)
    drugs_targets = get_target_verdicts(DRUGS_TARGET)
    
    print(f"   📊 Loaded {len(drugs_indictment)} indictment facts")
    print(f"   📊 Loaded citations for {len(drugs_citations)} verdicts")
    print(f"   📊 Loaded {len(drugs_sentencing)} sentencing ranges")
    print(f"   📊 Found {len(drugs_targets)} target verdicts")
    
    # Create unified file
    drugs_unified = create_unified_file(
        domain="drugs",
        indictment_facts_df=drugs_indictment,
        citations_dict=drugs_citations,
        sentencing_df=drugs_sentencing,
        target_verdicts=drugs_targets,
        output_path=OUTPUT_DIR / "drugs_unified.csv"
    )
    
    # Create tagging samples
    create_citations_for_tagging(
        drugs_citations, "drugs",
        OUTPUT_DIR / "tagging" / "drugs_citations_for_tagging.csv",
        sample_size=200
    )
    
    create_indictment_facts_for_llm_judge(
        drugs_indictment, "drugs",
        OUTPUT_DIR / "tagging" / "drugs_indictment_for_llm_judge.csv",
        sample_size=100
    )
    
    # ========== WEAPON DOMAIN ==========
    print("\n" + "=" * 40)
    print("🔫 WEAPON DOMAIN")
    print("=" * 40)
    
    # Load data
    weapon_indictment = load_indictment_facts(WEAPON_INDICTMENT_FACTS)
    weapon_citations = load_citations(WEAPON_CITATIONS_DIR)
    
    # Load weapon sentencing - use GPT extracted data (most complete)
    weapon_sentencing_gpt = load_sentencing_range(SENTENCING_RANGE_WEAPON)
    weapon_sentencing_gt = load_sentencing_range(SENTENCING_RANGE_WEAPON_GT, is_weapon_gt=True)
    
    # Use GPT extracted data (1394 POSITIVE), merge with GT for any missing
    if not weapon_sentencing_gpt.empty:
        weapon_sentencing = weapon_sentencing_gpt
        print(f"   📌 Using GPT extracted sentencing data ({len(weapon_sentencing_gpt)} verdicts)")
        # Optionally merge GT data for verdicts not in GPT results
        if not weapon_sentencing_gt.empty:
            gt_verdicts = set(weapon_sentencing_gt['verdict'])
            gpt_verdicts = set(weapon_sentencing_gpt['verdict'])
            missing_in_gpt = gt_verdicts - gpt_verdicts
            if missing_in_gpt:
                gt_extra = weapon_sentencing_gt[weapon_sentencing_gt['verdict'].isin(missing_in_gpt)]
                weapon_sentencing = pd.concat([weapon_sentencing, gt_extra], ignore_index=True)
                print(f"   📌 Added {len(gt_extra)} verdicts from GT data")
    elif not weapon_sentencing_gt.empty:
        weapon_sentencing = weapon_sentencing_gt
        print(f"   📌 Using GT sentencing data ({len(weapon_sentencing_gt)} verdicts)")
    else:
        weapon_sentencing = pd.DataFrame()
        print(f"   ⚠️ No weapon sentencing data found!")
    
    weapon_targets = get_target_verdicts(WEAPON_TARGET)
    
    print(f"   📊 Loaded {len(weapon_indictment)} indictment facts")
    print(f"   📊 Loaded citations for {len(weapon_citations)} verdicts")
    print(f"   📊 Loaded {len(weapon_sentencing)} sentencing ranges")
    print(f"   📊 Found {len(weapon_targets)} target verdicts")
    
    # Create unified file
    weapon_unified = create_unified_file(
        domain="weapon",
        indictment_facts_df=weapon_indictment,
        citations_dict=weapon_citations,
        sentencing_df=weapon_sentencing,
        target_verdicts=weapon_targets,
        output_path=OUTPUT_DIR / "weapon_unified.csv"
    )
    
    # Create tagging samples
    create_citations_for_tagging(
        weapon_citations, "weapon",
        OUTPUT_DIR / "tagging" / "weapon_citations_for_tagging.csv",
        sample_size=200
    )
    
    create_indictment_facts_for_llm_judge(
        weapon_indictment, "weapon",
        OUTPUT_DIR / "tagging" / "weapon_indictment_for_llm_judge.csv",
        sample_size=100
    )
    
    # ========== 5K DOMAIN ==========
    print("\n" + "=" * 40)
    print("📊 5K DOMAIN (Mixed cases)")
    print("=" * 40)
    
    # Load data
    five_k_indictment = load_indictment_facts(FIVE_K_INDICTMENT_FACTS)
    five_k_citations = load_citations(FIVE_K_CITATIONS_DIR)
    five_k_sentencing = load_sentencing_range(SENTENCING_RANGE_5K)
    
    print(f"   📊 Loaded {len(five_k_indictment)} indictment facts")
    print(f"   📊 Loaded citations for {len(five_k_citations)} verdicts")
    print(f"   📊 Loaded {len(five_k_sentencing)} sentencing ranges")
    
    # Create unified file (no target filtering for 5k)
    five_k_unified = create_unified_file(
        domain="5k",
        indictment_facts_df=five_k_indictment,
        citations_dict=five_k_citations,
        sentencing_df=five_k_sentencing,
        target_verdicts=set(),  # No filtering
        output_path=OUTPUT_DIR / "5k_unified.csv"
    )
    
    # ========== COMBINED FILE ==========
    print("\n" + "=" * 40)
    print("📁 CREATING COMBINED FILE")
    print("=" * 40)
    
    combined = pd.concat([drugs_unified, weapon_unified, five_k_unified], ignore_index=True)
    combined_path = OUTPUT_DIR / "all_domains_unified.csv"
    combined.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Saved combined file with {len(combined)} verdicts to {combined_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in OUTPUT_DIR.rglob("*.csv"):
        print(f"  - {f.relative_to(OUTPUT_DIR)}")
    
    print("\n📌 Next steps:")
    print("  1. Review drugs_unified.csv and weapon_unified.csv")
    print("  2. Use tagging files for manual annotation")
    print("  3. Run LLM as judge on indictment facts samples")
    

if __name__ == "__main__":
    main()

