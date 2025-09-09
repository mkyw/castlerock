#!/usr/bin/env python3
"""
Generate test cases (questions + reference answers + optional context snippets) from a website.

Usage:
  python scripts/generate_test_cases.py --url https://example.com --count 2 --out scripts/test_cases.json --mode append

Requirements:
  - GOOGLE_API_KEY in environment
  - beautifulsoup4 installed (pip install beautifulsoup4)
  - httpx installed
  - google-generativeai installed (pip install google-generativeai)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup (beautifulsoup4) not installed. Install with: pip install beautifulsoup4")
    raise

# Add repo root to path if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Load .env from project root (and also default .env resolution as fallback)
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

# Google Generative AI (Gemini)
try:
    import google.generativeai as genai
except ImportError:
    print("google-generativeai not installed. Install with: pip install google-generativeai")
    raise

DEFAULT_OUT = REPO_ROOT / "scripts" / "test_cases.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality question and answer pairs "
    "from provided source text. Return strict JSON only, no prose."
)

USER_PROMPT_TEMPLATE = (
    "Pretend you are a student at UW Madison. Based strictly on the source text below, generate {count} plausible Q&A pairs that a student would ask a customer service representative.\n\n"
    "Return the output as a valid JSON list. Each item must contain:\n"
    "- \"question\": A clear, well-formed question that can be answered using the source. Questions should test meaningful understanding, not trivial details.\n"
    "- \"reference_answer\": A concise, accurate, and self-contained answer based only on the source text. Answers must be factually correct, helpful, and phrased in complete sentences.\n"
    "\n"
    "Additional requirements:\n"
    "- Only use information explicitly in the source text; do not invent details.\n"
    "- Ensure variety in question types (who, what, whpythoy, how, etc.).\n"
    "- Each answer should be complete and useful on its own, not requiring extra context.\n"
    "\n"
    "Source text:\n" 
    """{source}"""
)

async def fetch_page_text(url: str, timeout: int = 20) -> str:
    async with httpx.AsyncClient(timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (compatible; TestCaseBot/1.0)"
    }) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Normalize whitespace
    text = " ".join(text.split())
    return text

async def generate_qa_from_text(source_text: str, count: int = 2, model: str = "gemini-1.5-flash") -> List[Dict[str, Any]]:
    import asyncio

    user_prompt = USER_PROMPT_TEMPLATE.format(count=count, source=source_text)

    def _call_gemini() -> str:
        # Configure per-call in case not already configured
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment.")
        genai.configure(api_key=api_key)

        model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model_obj.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1200,
                # Ask Gemini to return strict JSON
                "response_mime_type": "application/json",
            },
        )
        # The SDK surfaces text content as .text
        return getattr(response, "text", "")

    content = await asyncio.to_thread(_call_gemini)
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Model did not return a JSON list")
        # Basic schema cleanup
        cleaned: List[Dict[str, Any]] = []
        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("reference_answer") or "").strip()
            if q and a:
                cleaned.append({
                    "question": q,
                    "reference_answer": a,
                })
        return cleaned
    except Exception as e:
        # Try to salvage JSON by finding the first and last brackets
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            data = json.loads(content[start:end])
            if isinstance(data, list):
                return data
        except Exception:
            pass
        raise ValueError(f"Failed to parse model JSON: {e}\nRaw content:\n{content}")

def read_existing(out_path: Path) -> List[Dict[str, Any]]:
    if out_path.exists():
        try:
            with out_path.open("r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def write_cases(out_path: Path, cases: List[Dict[str, Any]], mode: str = "append") -> None:
    if mode not in {"append", "overwrite"}:
        raise ValueError("mode must be 'append' or 'overwrite'")
    if mode == "append":
        existing = read_existing(out_path)
        merged = existing + cases
    else:
        merged = cases
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {len(cases)} cases to {out_path} (mode={mode}). Total: {len(merged)}")

async def main():
    parser = argparse.ArgumentParser(description="Generate test cases from a website")
    parser.add_argument("--url", required=True, help="Website URL to generate questions from")
    parser.add_argument("--count", type=int, default=2, help="Number of Q&A pairs to generate")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output JSON path")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append", help="Append or overwrite output file")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters of source text to send to the model")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model to use (e.g., gemini-1.5-flash or gemini-1.5-pro)")

    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set in environment.")
        sys.exit(1)

    url = args.url
    text = await fetch_page_text(url)
    if len(text) > args.max_chars:
        text = text[: args.max_chars]

    cases = await generate_qa_from_text(text, count=args.count, model=args.model)
    out_path = Path(args.out)
    write_cases(out_path, cases, mode=args.mode)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
