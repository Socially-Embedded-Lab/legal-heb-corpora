import os
import re
import pandas as pd
from docx import Document as load_doc
from docx.document import Document as DocxDocument
# import stanza
import sys
from typing import Optional
import yaml

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)


from docx.text.paragraph import Paragraph
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl

# Modify property of Paragraph.text to include hyperlink text

Paragraph.text = property(lambda self: get_paragraph_text(self))

# Optional: Word COM automation for .doc -> .docx conversion on Windows
try:
    import win32com.client as win32  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    win32 = None

def get_paragraph_text(paragraph) -> str:
    """
    Extract text from paragraph, including hyperlink text.
    """
    def get_xml_tag(element):
        return "%s:%s" % (element.prefix, re.match("{.*}(.*)", element.tag).group(1))

    text_content = ''
    run_count = 0
    for child in paragraph._p:
        tag = get_xml_tag(child)
        if tag == "w:r":
            text_content += paragraph.runs[run_count].text
            run_count += 1
        if tag == "w:hyperlink":
            for sub_child in child:
                if get_xml_tag(sub_child) == "w:r":
                    text_content += sub_child.text
    return text_content

def is_paragraph_visually_bold(paragraph):
    """
    Check if the ENTIRE paragraph is bold or underlined.
    Returns True only if all meaningful text in the paragraph has bold/underline formatting.
    Includes support for complex scripts (Hebrew, Arabic, etc.) via w:bCs.
    """
    if not paragraph.runs or not paragraph.text.strip():
        return False
    
    total_text_length = 0
    formatted_text_length = 0
    
    # Method 1: Check individual runs for bold or underline
    for run in paragraph.runs:
        run_text = run.text.strip()
        if not run_text:  # Skip empty runs
            continue
        
        total_text_length += len(run_text)
        is_formatted = False
        
        # Check python-docx properties
        if run.bold or run.underline:
            is_formatted = True
        else:
            # Check XML-level bold (both regular and complex script)
            try:
                rpr = run._r.rPr
                if rpr is not None:
                    # Check for regular bold (w:b)
                    b = rpr.find('.//{*}b')
                    if b is not None and b.get('{*}val') != '0':
                        is_formatted = True
                    
                    # Check for complex script bold (w:bCs) - IMPORTANT for Hebrew!
                    if not is_formatted:
                        bCs = rpr.find('.//{*}bCs')
                        if bCs is not None and bCs.get('{*}val') != '0':
                            is_formatted = True
                    
                    # Check for underline element
                    if not is_formatted:
                        u = rpr.find('.//{*}u')
                        if u is not None:
                            is_formatted = True
            except:
                pass
        
        if is_formatted:
            formatted_text_length += len(run_text)
    
    # If no text found, return False
    if total_text_length == 0:
        return False
    
    # Calculate ratio of formatted text
    ratio = formatted_text_length / total_text_length
    
    # Method 2: Check paragraph style for bold formatting
    # (this applies to all text if no run-level formatting exists)
    if ratio < 0.9:  # Only check style if runs didn't show enough formatting
        try:
            style = paragraph.style
            if style is not None:
                if hasattr(style, '_element') and hasattr(style._element, 'rPr'):
                    rpr = style._element.rPr
                    if rpr is not None:
                        b = rpr.find('.//{*}b')
                        if b is not None and b.get('{*}val') != '0':
                            return True
                        
                        bCs = rpr.find('.//{*}bCs')
                        if bCs is not None and bCs.get('{*}val') != '0':
                            return True
                        
                        u = rpr.find('.//{*}u')
                        if u is not None:
                            return True
        except:
            pass
    
    # Method 3: Check paragraph properties directly
    if ratio < 0.9:
        try:
            pPr = paragraph._p.pPr
            if pPr is not None:
                rpr = pPr.find('.//{*}rPr')
                if rpr is not None:
                    b = rpr.find('.//{*}b')
                    if b is not None and b.get('{*}val') != '0':
                        return True
                    
                    bCs = rpr.find('.//{*}bCs')
                    if bCs is not None and bCs.get('{*}val') != '0':
                        return True
                    
                    u = rpr.find('.//{*}u')
                    if u is not None:
                        return True
        except:
            pass
    
    # Return True only if 90%+ of text is formatted
    # (allows for small inconsistencies like spaces or punctuation)
    return ratio >= 0.9

def is_block_bold(block) -> bool:
    """
    Check if the entire block/paragraph text is bold.
    """
    if block.runs:
        for run in block.runs:
            if run.bold or run.underline:
                return True
    return False


def is_paragraph_miriam_font(paragraph) -> bool:
    """
    Check if the paragraph uses Miriam font (or variants like Miriam Fixed, MiriamCLM).
    Returns True if any run in the paragraph uses a Miriam font.
    """
    if not paragraph.runs:
        return False
    
    for run in paragraph.runs:
        run_text = run.text.strip()
        if not run_text:
            continue
        
        # Check python-docx font property
        if run.font and run.font.name:
            font_name = run.font.name.lower()
            if 'miriam' in font_name:
                return True
        
        # Check XML-level font (for complex scripts like Hebrew)
        try:
            rpr = run._r.rPr
            if rpr is not None:
                # Check w:rFonts element for various font attributes
                rFonts = rpr.find('.//{*}rFonts')
                if rFonts is not None:
                    # Check all font attributes: ascii, hAnsi, cs (complex script), eastAsia
                    for attr in ['ascii', 'hAnsi', 'cs', 'eastAsia', 
                                 '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii',
                                 '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hAnsi',
                                 '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cs']:
                        font_val = rFonts.get(attr)
                        if font_val and 'miriam' in font_val.lower():
                            return True
                    # Also check attrib dict directly
                    for val in rFonts.attrib.values():
                        if val and 'miriam' in val.lower():
                            return True
        except:
            pass
    
    return False

def iterate_block_items(parent):
    """Yield top-level paragraphs in the document; skip most tables.

    Special case: if a table has exactly one cell and its text contains
    'גזר דין', yield that cell's paragraphs instead of skipping.
    """
    if isinstance(parent, DocxDocument):
        parent_element = parent.element.body
    elif isinstance(parent, _Cell):
        parent_element = parent._tc
    else:
        raise ValueError("Unsupported parent type.")

    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)

        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)

            # Check if table is a single cell
            if len(table.rows) == 1 and len(table.columns) == 1:
                cell = table.cell(0, 0)
                cell_text = cell.text or ""

                # If the single cell contains 'גזר דין', yield its paragraphs
                if "גזר דין" in cell_text:
                    for p in cell.paragraphs:
                        yield p
                    continue  # go to next child

            # Otherwise: skip this table entirely
            continue

def extract_part_after_number_or_hebrew_letter(sentence: str) -> str:
    """
    Extract text following a pattern of number or Hebrew letter.
    """
    pattern = r'^(?:[0-9\u05D0-\u05EA]+)\.\s*(.*)'
    match = re.search(pattern, sentence)
    return match.group(1).strip() if match else sentence
def doc_to_csv(doc_path: str = None, result_path: str = None):
    """
    Converts a DOCX document to a CSV where each row is a paragraph (not sentences),
    associating paragraphs with their most recent bold heading ("part").

    Parameters:
    - doc_path (str, optional): The path to the DOCX document. Defaults to None.

    Steps:
    1. Initialize data dictionary to hold extracted content.
    2. Open and iterate through the provided DOCX document.
    3. Filter out unnecessary blocks.
    4. Determine if the current block is a title or content.
    5. If it's content, tokenize it using the Stanza library.
    6. Add the extracted content to the data dictionary.
    7. Convert the data dictionary to a Pandas DataFrame.

    Returns:
    - DataFrame: A Pandas DataFrame containing the extracted text from the DOCX document with columns 'text' and 'part'.
    """

    data = {'verdict': [],'text': [], 'part': []}
    data['verdict']=os.path.splitext(os.path.basename(doc_path))[0]
    doc = load_doc(doc_path)
    part = 'nothing'  
    skip_mini_ratio = False

    for block in iterate_block_items(doc): # Updated usage
        flag = False
        # Skip paragraphs that are inside a table (parent XML tag w:tc)
        try:
            if getattr(block, "_p", None) is not None:
                parent = block._p.getparent()
                if hasattr(parent, "tag") and str(parent.tag).endswith('}tc'):
                    text = (block.text or "").strip()
                    if "גזר דין" not in text:
                        continue
                    else:
                        flag = True
        except Exception:
            # If detection fails for any reason, fall back to processing
            pass
        block_text = block.text
        if block_text =="גזר דין" or block_text == "גזר-דין":
            block_text = "רקע"
        

            # Detect מיני-רציו section start (even if not bold)
        if "מיני-רציו" in block_text or "מיני רציו" in block_text or "מיני - רציו" in block_text:
            skip_mini_ratio = True
            continue
        if re.search(r'(זכות ערעור|גביית קנסות|העתק לשירות המבחן|ניתן והודע היום|ניתן היום|ניתנה היום|העתק(?:\s+של)?\s+גזר(?:\s|-)?ה?דין)', block_text):
            break
        # Skip all lines under מיני-רציו until next bold title
        if skip_mini_ratio:
            if is_paragraph_visually_bold(block):   # new section started
                skip_mini_ratio = False
            else:
                continue
        # if len(block_text) <= 1 or 'ע"פ' in block_text[:20] or 'ת"פ' in block_text or 'עפ"ג' in block_text or "ע.פ" in block_text:
        #     continue
        if len(block_text) <= 1:
            continue

        # Check if this is a title/part: bold OR Miriam font, with additional conditions
        is_title = (is_paragraph_visually_bold(block) or is_paragraph_miriam_font(block)) and \
                   len(block_text.split(' ')) < 15 and \
                   "מדינת ישראל " not in block_text and \
                   "חודשי מאסר" not in block_text and \
                   "שנות מאסר" not in block_text and \
                   "מאסר בפועל" not in block_text and \
                   "זכות ערעור" not in block_text
        
        if is_title:
            part = block_text
            normalized_part = part.strip().replace("־", "-")
            if normalized_part in ["מיני-רציו", "מיני רציו", "מיני - רציו"]:
                continue
        else:
            # Split by paragraph: take the whole paragraph content (optionally strip leading numbering)
            text = extract_part_after_number_or_hebrew_letter(block_text).strip()
            # Skip quoted-only paragraphs heuristically
            if text.startswith('"') and (text.endswith('".') or text.endswith('"')):
                continue
            if text == part:
                continue
            if len(text.split(' ')) > 3:
                data['text'].append(text)
                data['part'].append(part)

    sentence_doc_df = pd.DataFrame(data)

    #remove duplicates text
    sentence_doc_df = sentence_doc_df.drop_duplicates(subset=['text'])
    #remove last 2 rows
    # sentence_doc_df = sentence_doc_df[:-2]

    # Decide output path: allow result_path to be a file path (..csv) or a directory
    if result_path and result_path.lower().endswith('.csv'):
        out_path = result_path
    else:
        if result_path is None:
            out_path = os.path.splitext(doc_path)[0] + '.csv'
        else:
            out_path = os.path.join(result_path, 'preprocessing_2.csv')

    sentence_doc_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path



def convert_doc_to_docx(doc_path: str, docx_path: str) -> None:
    """Convert a legacy .doc file to .docx using Microsoft Word COM (Windows only).

    Requires pywin32 (win32com). If not available, raises RuntimeError.
    """
    if win32 is None:
        raise RuntimeError("pywin32 (win32com) is not installed; cannot convert .doc to .docx")
    word = None
    try:
        word = win32.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(doc_path))
        # 16 == wdFormatXMLDocument
        doc.SaveAs(os.path.abspath(docx_path), FileFormat=16)
        doc.Close(False)
    finally:
        if word is not None:
            word.Quit()


if __name__ == "__main__":
    # read config yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from src/ to project root

    # Load configuration
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    params = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    types = params['TYPES']

    for type_case in types:
        # Resolve paths against the project root so this runs from any cwd
        folder_path = os.path.join(project_root, params['DOCX_PATH'].format(type=type_case))
        output_base = os.path.join(project_root, params['CSV_PATH'].format(type=type_case))
        os.makedirs(output_base, exist_ok=True)

        # Iterate files and process .doc/.docx
        for file in os.listdir(folder_path):
            # Skip temp/hidden files and folders
            if file.startswith('~$') or file.startswith('.'):
                continue
            in_path = os.path.join(folder_path, file)
            if not os.path.isfile(in_path):
                continue

            name, ext = os.path.splitext(file)
            ext = ext.lower()
            if file.startswith('m01'):
                continue
            # if file != "ME-10-08-17917-313.docx":
            #     continue
            # If .doc, try convert to .docx first
            if ext == ".doc":
                target_docx = os.path.join(folder_path, f"{name}.docx")
                if not os.path.exists(target_docx):
                    try:
                        print(f"Converting .doc -> .docx: {file}")
                        convert_doc_to_docx(in_path, target_docx)
                    except Exception as e:
                        print(f"[WARN] Skipping {file}: cannot convert .doc to .docx ({e})")
                        continue
                in_path = target_docx
                ext = ".docx"

            # Process .docx files
            if ext == ".docx":
                out_csv = os.path.join(output_base, f"{name}.csv")
                # check if output already exists
                if os.path.exists(out_csv):
                    print(f"Skipping {file}: output CSV already exists.")
                    continue
                try:
                    print(f"Processing {os.path.basename(in_path)} -> {out_csv}")
                    written = doc_to_csv(in_path, out_csv)
                    print(f"Saved sentences -> {written}")
                except Exception as e:
                    print(f"[ERROR] Failed processing {in_path}: {e}")
            else:
                # Not a Word document
                continue