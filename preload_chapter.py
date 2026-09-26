"""
One-shot script: seeds TaleTrace's merge memory with a chapter of text
so that intent-button recaps work from the very first tap during a demo.

Usage:
    python preload_chapter.py              # Load demo_chapter.txt into memory
    python preload_chapter.py --reset      # Wipe memory, then load
    python preload_chapter.py --reset-only # Wipe memory, don't load anything

What it does:
    1. Reads demo_chapter.txt from the project root.
    2. Calls update_memory_and_pointer_map() — the same function main.py uses.
    3. This populates reading_memory.json, pointer_map.json, and reading_state.py.
    4. When you run main.py and tap intent, recap_up_to() will have text to summarise.

NOTE: When merge_memory is empty, pipeline.process_ocr_text() calls
      merge_ocr_with_groq("", chapter_text) which returns the text directly
      — NO Groq API call is made. The preload is 100% offline.
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 console output on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def reset_memory() -> None:
    """Wipe all persistent memory files and reset the in-memory pipeline."""
    from ocr_memory import pipeline

    files_to_clear = [
        Path("reading_memory.json"),
        Path("pointer_map.json"),
        Path("reading_state.py"),
    ]
    for f in files_to_clear:
        if f.exists():
            f.unlink()
            print(f"  Deleted {f.name}")
        else:
            print(f"  {f.name} (not present, skip)")

    # Reset in-memory state
    pipeline.merge_memory = ""
    pipeline.current_reading_pointer = 0
    pipeline.current_word_index = 0
    print("  In-memory pipeline reset.")


def preload(chapter_path: Path) -> None:
    """Load chapter text into merge memory via the real pipeline."""
    from merge_engine import update_memory_and_pointer_map, get_memory_text

    if not chapter_path.exists():
        raise SystemExit(f"ERROR: {chapter_path} not found.")

    text = chapter_path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"ERROR: {chapter_path} is empty.")

    word_count = len(text.split())
    print(f"Loading {len(text):,} chars ({word_count:,} words) from {chapter_path.name}...")

    # This call:
    #   1. Calls pipeline.process_ocr_text(text) → merge_ocr_with_groq("", text) → returns text (no API call)
    #   2. Builds word pointer map
    #   3. Writes reading_state.py, reading_memory.json, pointer_map.json
    pointer_map = update_memory_and_pointer_map(text, current_speaking_word_index=0)

    total_words = len(pointer_map.get("words", []))
    mem = get_memory_text()
    print(f"\nMemory populated: {total_words:,} words indexed.")
    print(f"First 80 chars: {mem[:80]}...")
    print(f"Last 80 chars:  ...{mem[-80:]}")
    print(f"\nFiles written:")
    for f in ["reading_memory.json", "pointer_map.json", "reading_state.py"]:
        p = Path(f)
        if p.exists():
            print(f"  {f} ({p.stat().st_size:,} bytes)")
    print(f"\nReady. Run `python main.py` and tap intent — the recap will cover this chapter.")


def main():
    parser = argparse.ArgumentParser(description="Preload TaleTrace memory with demo chapter text.")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe all memory files before loading.")
    parser.add_argument("--reset-only", action="store_true",
                        help="Wipe all memory files and exit (don't load anything).")
    parser.add_argument("--file", type=str, default="demo_chapter.txt",
                        help="Path to chapter text file (default: demo_chapter.txt)")
    args = parser.parse_args()

    print("=" * 60)
    print("TaleTrace Demo Preloader")
    print("=" * 60)

    if args.reset or args.reset_only:
        print("\nResetting memory...")
        reset_memory()
        if args.reset_only:
            print("\nMemory wiped. Exiting.")
            return

    chapter_path = Path(args.file)
    print()
    preload(chapter_path)


if __name__ == "__main__":
    main()
