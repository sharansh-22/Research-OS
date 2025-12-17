"""
Quick test for your text extraction functions
Run this to see what's broken in your code
"""

import re
from typing import List, Dict

# ============================================================================
# YOUR ORIGINAL FUNCTIONS (Copy from your notebook)
# ============================================================================

def basic_clean(text: str) -> str:
    """Your original cleaning function"""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str):
    """Your original paragraph splitting"""
    raw_paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    return paragraphs


def build_chunks(
    paragraphs: List[str],
    max_words: int = 300,
    overlap_words: int = 50,
    source_name: str = "test.pdf"
) -> List[Dict]:
    """Your original chunking function"""
    chunks = []
    current_words = []
    
    def flush_current_chunk():
        nonlocal current_words
        if not current_words:
            return
        chunk_text = " ".join(current_words).strip()
        n_words = len(chunk_text.split())
        chunk = {
            "id": len(chunks),
            "text": chunk_text,
            "n_words": n_words,
            "source": source_name
        }
        chunks.append(chunk)
        current_words = []
    
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        
        while words:
            remaining_space = max_words - len(current_words)
            
            if remaining_space <= 0:
                flush_current_chunk()
                
                if overlap_words > 0 and chunks:
                    last_chunk_words = chunks[-1]["text"].split()
                    overlap = last_chunk_words[-overlap_words:]
                    current_words = overlap.copy()
                else:
                    current_words = []
                continue
            
            take = min(remaining_space, len(words))
            current_words.extend(words[:take])
            words = words[take:]
    
    if current_words:
        flush_current_chunk()
    
    return chunks


# ============================================================================
# TEST SUITE
# ============================================================================

def test_1_paragraph_detection():
    """TEST 1: Can it find paragraphs correctly?"""
    print("\n" + "="*70)
    print("TEST 1: Paragraph Detection")
    print("="*70)
    
    # Test case 1: Clear paragraphs (double newline)
    text1 = """First paragraph here.
More of first paragraph.

Second paragraph here.

Third paragraph here."""
    
    clean1 = basic_clean(text1)
    paras1 = split_into_paragraphs(clean1)
    
    print(f"\nTest Case 1: Text with clear paragraph breaks")
    print(f"  Expected: 3 paragraphs")
    print(f"  Got: {len(paras1)} paragraphs")
    print(f"  Result: {'✓ PASS' if len(paras1) == 3 else '✗ FAIL'}")
    
    # Test case 2: No paragraph breaks (like your terminal.pdf)
    text2 = """Line 1
Line 2
Line 3
Line 4
Line 5"""
    
    clean2 = basic_clean(text2)
    paras2 = split_into_paragraphs(clean2)
    
    print(f"\nTest Case 2: Text with single newlines (like terminal.pdf)")
    print(f"  Expected: 5 separate items OR 1 paragraph (depends on design)")
    print(f"  Got: {len(paras2)} paragraphs")
    print(f"  Result: {'⚠️  ISSUE' if len(paras2) == 1 else '✓ OK'}")
    if len(paras2) == 1:
        print(f"  Problem: Entire document treated as ONE paragraph!")
        print(f"  Impact: Creates fewer, larger chunks than expected")


def test_2_input_validation():
    """TEST 2: Does it validate bad inputs?"""
    print("\n" + "="*70)
    print("TEST 2: Input Validation")
    print("="*70)
    
    text = "word " * 100
    paras = split_into_paragraphs(basic_clean(text))
    
    bad_inputs = [
        (20, 30, "overlap > max_words"),
        (20, 20, "overlap = max_words"),
        (-10, 5, "negative max_words"),
        (10, -5, "negative overlap"),
        (0, 0, "zero values"),
    ]
    
    failures = 0
    for max_w, overlap_w, desc in bad_inputs:
        print(f"\n  Test: {desc} (max={max_w}, overlap={overlap_w})")
        try:
            chunks = build_chunks(paras, max_words=max_w, overlap_words=overlap_w)
            print(f"    ✗ FAIL: Should reject but created {len(chunks)} chunks")
            failures += 1
        except (ValueError, AssertionError) as e:
            print(f"    ✓ PASS: Correctly rejected with {type(e).__name__}")
        except Exception as e:
            print(f"    ⚠️  UNEXPECTED: {type(e).__name__}: {e}")
            failures += 1
    
    print(f"\n  Total validation failures: {failures}/5")


def test_3_chunk_overlap():
    """TEST 3: Is overlap working correctly?"""
    print("\n" + "="*70)
    print("TEST 3: Chunk Overlap")
    print("="*70)
    
    # Create exactly 50 words
    words = [f"word{i:02d}" for i in range(50)]
    text = " ".join(words)
    
    paras = split_into_paragraphs(basic_clean(text))
    chunks = build_chunks(paras, max_words=20, overlap_words=5)
    
    print(f"\n  Input: 50 words")
    print(f"  Config: max_words=20, overlap=5")
    print(f"  Generated: {len(chunks)} chunks")
    
    # Check each chunk
    for i, chunk in enumerate(chunks):
        chunk_words = chunk["text"].split()
        print(f"\n  Chunk {i}: {len(chunk_words)} words")
        print(f"    Starts: {chunk_words[0]} ... Ends: {chunk_words[-1]}")
        
        # Check if overlap works between consecutive chunks
        if i > 0:
            prev_chunk_words = chunks[i-1]["text"].split()
            prev_last_5 = prev_chunk_words[-5:]
            curr_first_5 = chunk_words[:5]
            
            if prev_last_5 == curr_first_5:
                print(f"    ✓ Overlap with previous chunk: {prev_last_5[:2]}...{prev_last_5[-1]}")
            else:
                print(f"    ✗ FAIL: No overlap detected!")
                print(f"       Previous ends: {prev_last_5}")
                print(f"       Current starts: {curr_first_5}")


def test_4_empty_input():
    """TEST 4: Can it handle empty input?"""
    print("\n" + "="*70)
    print("TEST 4: Empty Input Handling")
    print("="*70)
    
    test_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("\n\n\n", "Newlines only"),
    ]
    
    for text, desc in test_cases:
        print(f"\n  Test: {desc}")
        try:
            clean = basic_clean(text)
            paras = split_into_paragraphs(clean)
            chunks = build_chunks(paras, max_words=10, overlap_words=2)
            
            if len(chunks) == 0:
                print(f"    ✓ PASS: Returns empty list ({len(chunks)} chunks)")
            else:
                print(f"    ⚠️  UNEXPECTED: Created {len(chunks)} chunks from empty input")
        except Exception as e:
            print(f"    ✗ FAIL: Crashed with {type(e).__name__}: {e}")


def test_5_real_world_simulation():
    """TEST 5: Simulate your terminal.pdf scenario"""
    print("\n" + "="*70)
    print("TEST 5: Real-World Scenario (Terminal Cheat Sheet)")
    print("="*70)
    
    # Simulate terminal.pdf structure
    terminal_text = """Linux & Terminal Command Cheat Sheet
1. File & Directory Navigation
pwd
ls
ls -l
cd folder
cd ..
2. File Management
touch file.txt
mkdir folder
rm file
cp file1 file2
mv old new
3. Permissions
chmod 755 file
chown user file
4. Search
find . -name "*.txt"
grep "text" file"""
    
    print(f"\n  Input: {len(terminal_text)} characters")
    print(f"  Structure: Title + sections with commands")
    
    clean = basic_clean(terminal_text)
    paras = split_into_paragraphs(clean)
    chunks = build_chunks(paras, max_words=50, overlap_words=10)
    
    print(f"\n  Results:")
    print(f"    Paragraphs detected: {len(paras)}")
    print(f"    Chunks created: {len(chunks)}")
    
    if len(paras) == 1:
        print(f"\n  ⚠️  PROBLEM IDENTIFIED:")
        print(f"     Entire document treated as single paragraph!")
        print(f"     This means all {terminal_text.count(' ')} words in one paragraph")
    
    print(f"\n  Chunk breakdown:")
    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"    Chunk {i}: {chunk['n_words']} words - '{preview}...'")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and show summary"""
    print("\n" + "="*70)
    print("           TESTING YOUR PDF EXTRACTION CODE")
    print("="*70)
    print("This will reveal issues in your current implementation")
    print("="*70)
    
    test_1_paragraph_detection()
    test_2_input_validation()
    test_3_chunk_overlap()
    test_4_empty_input()
    test_5_real_world_simulation()
    
    print("\n" + "="*70)
    print("                    TEST SUMMARY")
    print("="*70)
    print("\nExpected Issues:")
    print("  1. ✗ No input validation (accepts invalid parameters)")
    print("  2. ✗ Terminal.pdf becomes 1 giant paragraph (poor chunking)")
    print("  3. ⚠️  Overlap might not work as expected")
    print("  4. ✓ Empty input probably handled OK")
    print("\nThese are NORMAL issues - we'll fix them in Step 4!")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()


    