#!/usr/bin/env python3
"""
RAG Testing Script

This script compares the performance of RAG (Retrieval-Augmented Generation) vs 
non-retrieval LLM approaches using predefined questions and answers.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv  # type: ignore
    # Load project-level .env so backend and baseline share keys
    load_dotenv(dotenv_path=(Path(__file__).resolve().parent.parent / ".env"), override=False)
except Exception:
    pass
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
import httpx
from pathlib import Path

# Add the parent directory to sys.path to import from backend
sys.path.append(str(Path(__file__).parent.parent))

# Import from backend
from backend.kb_rag_system import KBScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default test questions and expected answers
DEFAULT_TEST_CASES = [
    {
        "question": "What is RAG?",
        "expected_answer_keywords": ["retrieval", "augmented", "generation", "documents", "context"],
        "context": "RAG stands for Retrieval-Augmented Generation, a technique that enhances LLM responses by retrieving relevant documents and using them as context."
    },
    {
        "question": "How does vector search work?",
        "expected_answer_keywords": ["embeddings", "vectors", "similarity", "cosine", "distance"],
        "context": "Vector search works by converting text into numerical vectors (embeddings) and finding similar vectors using distance metrics like cosine similarity."
    },
    {
        "question": "What are the benefits of using RAG over standard LLMs?",
        "expected_answer_keywords": ["up-to-date", "information", "hallucination", "grounding", "context"],
        "context": "RAG provides benefits like reducing hallucinations, providing up-to-date information, and grounding responses in specific knowledge sources."
    }
]

class RAGTester:
    """Class to test and compare RAG vs non-retrieval LLM performance"""
    
    def __init__(
        self, 
        index_name: Optional[str] = None,
        user_id: str = "tester",
        test_cases: Optional[List[Dict[str, Any]]] = None,
        output_file: Optional[str] = None,
        seed_context: bool = False,
        resume: bool = False,
        history_path: Optional[str] = None,
    ):
        """
        Initialize the RAG tester
        
        Args:
            index_name: Name of the Pinecone index to use for RAG
            user_id: User ID for the KBScraper
            test_cases: List of test cases with questions and expected answers
            output_file: Path to save test results
        """
        self.index_name = index_name
        self.user_id = user_id
        self.test_cases = test_cases or DEFAULT_TEST_CASES
        self.output_file = output_file or f"rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.seed_context = seed_context
        self.resume = resume
        self.history_path = history_path or str(Path("scripts/results/history.jsonl"))
        
        # Initialize RAG system
        self.rag_system = None
        
        # Results storage
        self.results = {
            "index_name": index_name,
            "test_cases": [],
            "summary": {
                "total_questions": len(self.test_cases),
                "rag_similarity": 0.0,
                "llm_similarity": 0.0,
                "similarity_lift": 0.0,
                # Counts of low-scoring answers (< 0.2)
                "rag_below_0_2": 0,
                "llm_below_0_2": 0,
            }
        }

        # Resume from existing output file if requested
        try:
            if self.resume and self.output_file and Path(self.output_file).exists():
                with open(self.output_file, 'r') as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and "test_cases" in existing:
                    self.results = existing
                    # Ensure summary has total_questions set to current test suite length
                    self.results.setdefault("summary", {})
                    self.results["summary"]["total_questions"] = len(self.test_cases)
                    print(f"[Resume] Loaded {len(self.results['test_cases'])} completed results from {self.output_file}")
        except Exception as e:
            logger.warning(f"Resume load failed: {e}")
    
    async def initialize_rag(self):
        """Initialize the RAG system"""
        logger.info(f"Initializing RAG system with index: {self.index_name}")
        print(f"[Init] Initializing RAG system (index={self.index_name})")
        self.rag_system = KBScraper(user_id=self.user_id, index_name=self.index_name)
        
        # Optionally seed per-test contexts into the RAG index for self-contained tests
        if self.seed_context:
            print("[Init] Seeding per-test contexts into the index...")
            for test_case in self.test_cases:
                if "context" in test_case and test_case["context"]:
                    # Create a unique URL for each context
                    url = f"test://context/{hash(test_case['question'])}"
                    await self.rag_system.process_document(url, test_case["context"])
            print("[Init] Context seeding complete.")
    
    async def query_rag(self, question: str) -> str:
        """Query the RAG system for an answer to a question."""
        print(f"[RAG] Querying RAG with question: {question}")
        # Ensure RAG is initialized
        try:
            if self.rag_system is None:
                await self.initialize_rag()
        except Exception as e:
            logger.warning(f"[RAG] Initialization failed: {e}")
            # proceed to retry loop; backend may become available later
        import asyncio as _asyncio
        attempt = 0
        while True:
            attempt += 1
            # Throttle API calls
            try:
                await _asyncio.sleep(2)
            except Exception:
                pass
            try:
                result = await self.rag_system.query(question)
                if result is None:
                    logger.warning(f"[RAG] Backend returned None result; retrying (attempt {attempt})")
                    continue
                if not isinstance(result, dict):
                    logger.warning(f"[RAG] Unexpected result type: {type(result)}; retrying (attempt {attempt})")
                    continue
                status = (result.get("status") or "success").lower()
                if status != "success":
                    err = result.get("answer") or result.get("error") or "unknown error"
                    print(f"[RAG] Warning: RAG returned status='{status}': {err} (attempt {attempt})")
                    continue
                answer = (result.get("answer") or "").strip()
                if answer:
                    return answer
                logger.warning(f"[RAG] Empty answer received; retrying (attempt {attempt})")
                continue
            except Exception as e:
                # This catches errors like "'NoneType' object has no attribute 'get'" thrown inside backend
                logger.warning(f"RAG query failed: {e} (attempt {attempt})")
                continue
    
    async def query_llm(self, question: str) -> str:
        """Query the LLM without retrieval"""
        # Use the same LLM as in RAG but without retrieval context
        # Prefer Gemini if key is present, otherwise fallback to OpenAI
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
            except ImportError:
                print("[LLM] google-generativeai not installed. Install with: pip install google-generativeai")
                return ""
            else:
                print(f"[BASELINE][LLM] Querying Gemini with question: {question}")
                genai.configure(api_key=gemini_key)
                # Use same default model as backend; allow override via env
                gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
                model = genai.GenerativeModel(gemini_model_name)
                print(f"[BASELINE][LLM] Gemini model: {gemini_model_name}")
                # Safety settings are optional; some SDK versions expect enum-mapped dicts.
                # To avoid KeyErrors like 'harm_category_sexual_content', default to None.
                baseline_safety_settings = None
                # Gemini SDK is sync; call it in a worker thread to avoid blocking the event loop
                import asyncio as _asyncio
                def _gen_content():
                    baseline_prompt = (
                        f"If you are not certain of the answer, do not guess, and only reply with 'I do not know.'{question}"
                    )
                    kwargs = {
                        "generation_config": {
                            # "temperature": 0.2,
                            "max_output_tokens": 4096,
                            "response_mime_type": "text/plain",
                            "candidate_count": 1,
                        }
                    }
                    if baseline_safety_settings is not None:
                        kwargs["safety_settings"] = baseline_safety_settings
                    return model.generate_content(baseline_prompt, **kwargs)
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        # Throttle API calls (async-friendly)
                        await _asyncio.sleep(2)
                        resp = await _asyncio.to_thread(_gen_content)
                        # Extract content robustly
                        answer = ""
                        try:
                            candidates = getattr(resp, "candidates", None)
                            if candidates:
                                for c in candidates:
                                    content = getattr(c, "content", None)
                                    parts = getattr(content, "parts", []) if content else []
                                    for p in parts:
                                        t = getattr(p, "text", None)
                                        if isinstance(t, str) and t.strip():
                                            answer += t
                        except Exception:
                            pass
                        # Do NOT access resp.text; it may raise when there are no Parts.
                        if not answer:
                            # optional diag when blocked
                            try:
                                pf = getattr(resp, "prompt_feedback", None)
                                if pf is not None and getattr(pf, "block_reason", None):
                                    print(f"[LLM][Gemini] blocked: {pf.block_reason}")
                            except Exception:
                                pass
                        answer = (answer or "").strip()
                        if answer:
                            return answer
                        print(f"[LLM][Gemini] Empty content. Retrying (attempt {attempt})")
                        continue
                    except Exception as e:
                        print(f"[LLM] Gemini error: {e} (attempt {attempt})")
                        continue
                
        # No gemini key configured: respect user request to avoid other providers
        print("[LLM] Gemini key not set; baseline LLM will be empty per config (no OpenAI fallback).")
        return ""
    
    def calculate_similarity(self, question: str, answer: str, reference_answer: str) -> float:
        """Minimal LLM-as-judge: send reference and candidate, get a numeric score [0,1]."""
        # Basic validation
        question = (question or "").strip()
        ref = (reference_answer or "").strip()
        ans = (answer or "").strip()
        if not ref or not ans:
            return 0.0

        # Require Gemini key
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            return 0.0

        # Lazy import and configure SDK, then cache model
        try:
            import google.generativeai as genai  # type: ignore
            # Import safety enums locally to avoid module-level dependency
            try:
                from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
            except Exception:
                HarmCategory = None  # type: ignore
                HarmBlockThreshold = None  # type: ignore
        except Exception:
            return 0.0
        try:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)  # ensure configured in this scope
        except Exception:
            return 0.0
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        if getattr(self, "_judge_model", None) is None or getattr(self, "_judge_model_name", None) != model_name:
            self._judge_model = genai.GenerativeModel(model_name)
            self._judge_model_name = model_name

        # Prepare rubric prompt (rubric-only as requested)
        rubric_prompt = (
            "You are an impartial evaluator. Your task is to grade a candidate answer to a given question against an ideal gold answer, using a numerical score from 1 to 1000.\n\n"
            "### Evaluation Criteria\n"
            "1. **Correctness (weight: highest)**  \n"
            "   - Does the candidate answer factually answer the question?  \n"
            "   - Is the answer free of any inaccuracies, hallucinations, or misleading statements?  \n"
            "2. **Accuracy of Content**  \n"
            "   An answer that has correct elements lost in a sea of irrelevant information is not useless."
            "   - Is the information grounded, precise, and relevant to the question?  \n"
            "   - Does it avoid adding irrelevant or speculative material?  \n"
            "3. **Helpfulness**  \n"
            "   A user needs to be able to easily fix their problems by directly following the instructions in the answer."
            "   - Does the answer directly address the question and provide sufficient detail?  \n"
            "   - Is the answer authoritative? Will users be confident in following the given advice?\n"
            "4. **Brevity & Conciseness**  \n"
            "   A long, winding answer is difficult to read and understand. Consider:"
            "   - Is the answer expressed clearly without unnecessary verbosity?  \n"
            "   - Does it avoid repetition or filler?  \n"
            "   - None of the reference answers are long. If the candidate answer is longer than reference, it should receive increasing penalties."
            "### Scoring Scale (1–1000)\n"
            "- **1000:** If an answer is short, the contents are completely correct, this is max score.\n"
            "- **900–1000:** Exceptional. Perfectly correct, accurate, concise, and helpful.  \n"
            "- **700–899:** Strong. Mostly correct and helpful, with only minor issues (slight verbosity or small omissions).  \n"
            "- **400–699:** Adequate. Partially correct but with noticeable issues (incompleteness, minor inaccuracies, lack of clarity).  \n"
            "- **200–399:** Weak. Mostly incorrect, unhelpful, or confusing.  \n"
            "- **1–199:** Very poor. Completely wrong, irrelevant, or misleading.  \n"
            "If the answer states 'I don't know' in some form or fashion, automatically award it 0 points and skip the instructions below."
            "### Output Formula\n"
            "First, consider what the question is asking. Then, compare the candidate answer to the gold answer.\n"
            "Finally, consider each of the stated criteria. If the answer to any of the rubric questions is not a resounding 'yes', points must be deducted. Evaluate how many points the candidate answer should gain or lose on each of them with the scoring scale in mind.\n"
            "Output only the numeric score (1–1000).  \n"
            "---\n"
            "**Question:** {question}  \n"
            "**Gold Answer:** {ref}  \n"
            "**Candidate Answer:** {ans}\n"
        )
        rubric_prompt = rubric_prompt.format(question=question, ref=ref, ans=ans)
        # Configure permissive safety to avoid empty outputs due to filtering
        safety_settings = None
        safety_mode = "default"
        try:
            if 'HarmCategory' in locals() and HarmCategory is not None and 'HarmBlockThreshold' in locals() and HarmBlockThreshold is not None:
                safety_settings = [
                    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                    {"category": HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                ]
                safety_mode = "custom-enum"
            else:
                # Fallback to string-based safety settings (some SDK versions accept these)
                safety_settings = [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                safety_mode = "custom-str"
        except Exception:
            safety_settings = None
            safety_mode = "default"
        # Single rubric call with infinite retry until non-empty numeric score
        # Use env override for output tokens if provided
        try:
            max_out = int(os.getenv("JUDGE_MAX_OUT", "8192"))
        except Exception:
            max_out = 8192
        import time as _time
        _dbg = os.getenv("JUDGE_DEBUG")
        while True:
            # Throttle API calls
            try:
                _time.sleep(2)
            except Exception:
                pass
            try:
                resp = self._judge_model.generate_content(
                    rubric_prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "text/plain",
                        "max_output_tokens": max_out,
                        "candidate_count": 1,
                    },
                    **({"safety_settings": safety_settings} if safety_settings is not None else {}),
                )
            except Exception as e:
                if _dbg:
                    print(f"[Judge][dbg] Exception during judge call: {e}")
                # retry
                continue

            if _dbg:
                try:
                    # Basic prompt stats
                    ref_head = (ref[:120]).replace("\n", " ")
                    cand_head = (ans[:120]).replace("\n", " ")
                    print(f"[Judge][dbg] model={self._judge_model_name} max_out={max_out} temp=0 safety={safety_mode}")
                    print(f"[Judge][dbg] prompt_len={len(rubric_prompt)} ref_len={len(ref)} ans_len={len(ans)}")
                    print(f"[Judge][dbg] ref_head='{ref_head}'…")
                    print(f"[Judge][dbg] cand_head='{cand_head}'…")
                    # Dump candidate finish reasons and part previews
                    cands = getattr(resp, 'candidates', []) or []
                    fins = [getattr(c, 'finish_reason', None) for c in cands]
                    print(f"[Judge][dbg] finish_reasons={fins}")
                    if cands:
                        for i, c in enumerate(cands):
                            parts = []
                            for p in getattr(c, 'content', {}).parts or []:
                                frag = getattr(p, 'text', None)
                                if frag:
                                    parts.append(f"len={len(frag)} {frag[:8]!r}…")
                            print(f"[Judge][dbg] cand#{i} parts={parts}")
                except Exception:
                    pass

            # Extract text
            text = None
            try:
                t = getattr(resp, "text", None)
                if isinstance(t, str) and t.strip():
                    text = t
            except Exception:
                text = ""

            if not text:
                try:
                    cands = getattr(resp, "candidates", None) or []
                    for c in cands:
                        content = getattr(c, "content", None)
                        for p in (getattr(content, "parts", []) if content else []):
                            t = getattr(p, "text", None)
                            if isinstance(t, str) and t.strip():
                                text += t
                except Exception:
                    text = ""

            if not text:
                if _dbg:
                    print("[Judge][dbg] RAW=<empty>")
                    # Extra diagnostics to understand why response is empty
                    try:
                        pf = getattr(resp, "prompt_feedback", None)
                        if pf is not None:
                            # Try to pull common fields if present
                            br = getattr(pf, "block_reason", None)
                            sr = getattr(pf, "safety_ratings", None)
                            print(f"[Judge][dbg] prompt_feedback.block_reason={br}")
                            print(f"[Judge][dbg] prompt_feedback.safety_ratings={sr}")
                        um = getattr(resp, "usage_metadata", None)
                        if um is not None:
                            print(f"[Judge][dbg] usage_metadata={um}")
                        # Candidate-level safety info
                        cands = getattr(resp, 'candidates', []) or []
                        for i, c in enumerate(cands):
                            fr = getattr(c, 'finish_reason', None)
                            sr = getattr(c, 'safety_ratings', None)
                            print(f"[Judge][dbg] cand#{i}.finish_reason={fr} cand#{i}.safety_ratings={sr}")
                    except Exception as _e:
                        try:
                            print(f"[Judge][dbg] empty_diag_err={_e}")
                        except Exception:
                            pass
                # retry the loop
                continue

            # Parse score in 0..1000 then scale to [0,1]
            s = (text or "").strip()
            if _dbg:
                try:
                    print(f"[Judge][dbg] RAW={s!r}")
                except Exception:
                    pass
            try:
                # Prefer the first number token
                # If the string begins with digits, parse that; otherwise search
                import re as _re
                m0 = _re.match(r"^\s*(\d{1,4}(?:\.\d+)?)", s)
                if m0:
                    score_val = float(m0.group(1))
                else:
                    raise ValueError("no leading number")
            except Exception:
                import re as _re
                # accept integers/floats between 0 and 1000
                m = _re.search(r"\b(?:1000|\d{1,3})(?:\.\d+)?\b", s)
                try:
                    score_val = float(m.group(0)) if m else None
                except Exception:
                    score_val = None
            if score_val is None:
                if _dbg:
                    print("[Judge][dbg] Could not parse numeric score; retrying")
                # retry
                continue
            # scale to [0,1]
            scaled = score_val / 1000.0
            return max(0.0, min(scaled, 1.0))
    
    async def run_tests(self):
        """Run the RAG tests"""
        # Determine starting index for resume
        start_idx = len(self.results.get("test_cases", [])) if self.resume else 0
        if start_idx:
            print(f"[Resume] Continuing from question {start_idx+1}")
        
        # Ensure RAG is initialized once before running tests
        try:
            if self.rag_system is None:
                await self.initialize_rag()
        except Exception as e:
            logger.warning(f"RAG initialize error (will continue and retry per-question): {e}")
        
        # Initialize totals from any existing results (resume)
        rag_total_similarity = sum(tc["rag_similarity"] for tc in self.results.get("test_cases", []))
        llm_total_similarity = sum(tc["llm_similarity"] for tc in self.results.get("test_cases", []))
        
        try:
            for i in range(start_idx, len(self.test_cases)):
                test_case = self.test_cases[i]
                question = test_case["question"]
                reference_answer = test_case["reference_answer"]
                
                logger.info(f"Testing question {i+1}/{len(self.test_cases)}: {question}")
                print(f"[Run] ({i+1}/{len(self.test_cases)}) Question: {question}")
                
                # Get answers from both systems
                rag_answer = await self.query_rag(question)
                llm_answer = await self.query_llm(question)

                # Print answers (truncated) for readability FIRST to make sequence obvious
                print(f"[Ans][RAG] {self._shorten(rag_answer)}")
                print(f"[Ans][LLM] {self._shorten(llm_answer)}")

                # Now calculate similarity using LLM judge
                rag_similarity = self.calculate_similarity(question, rag_answer, reference_answer)
                llm_similarity = self.calculate_similarity(question, llm_answer, reference_answer)

                # Update totals
                rag_total_similarity += rag_similarity
                llm_total_similarity += llm_similarity

                # Store results record
                rec = {
                    "question": question,
                    "reference_answer": reference_answer,
                    "rag_answer": rag_answer,
                    "llm_answer": llm_answer,
                    "rag_similarity": rag_similarity,
                    "llm_similarity": llm_similarity,
                    "similarity_lift": rag_similarity - llm_similarity,
                }
                self.results["test_cases"].append(rec)

                # Update and save incrementally
                self._update_summary_from_results()
                logger.info(f"  RAG similarity: {rag_similarity:.2f}, LLM similarity: {llm_similarity:.2f}")
                print(f"[Run]    RAG sim: {rag_similarity:.2f} | LLM sim: {llm_similarity:.2f} | Lift: {(rag_similarity-llm_similarity):.2f}")
                self.save_results()
                # Append history row
                self._append_history({
                    "index_name": self.index_name,
                    "question_idx": i + 1,
                    "rag_similarity": rag_similarity,
                    "llm_similarity": llm_similarity,
                    "similarity_lift": rec["similarity_lift"],
                })
        finally:
            # Ensure we at least persist the latest state on crash/interrupt
            try:
                self.save_results()
            except Exception as e:
                logger.warning(f"Final save failed: {e}")
        
        # Final summary log
        logger.info(f"Testing complete. Overall similarity lift: {self.results['summary']['similarity_lift']:.2f}")
        print(f"[Run] Testing complete. Overall lift: {self.results['summary']['similarity_lift']:.2f}")
        
        return self.results
    
    def save_results(self):
        """Save test results to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {self.output_file}")
    
    def _append_history(self, row: Dict[str, Any]):
        """Append a single JSON line to history file (creates directory if needed)."""
        try:
            hist_path = Path(self.history_path)
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hist_path, 'a') as hf:
                hf.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write history: {e}")
    
    def _update_summary_from_results(self):
        """Recompute running averages for summary from accumulated test_cases."""
        if not self.results.get("test_cases"):
            return
        rag_avg = sum(tc.get("rag_similarity", 0.0) for tc in self.results["test_cases"]) / len(self.results["test_cases"])
        llm_avg = sum(tc.get("llm_similarity", 0.0) for tc in self.results["test_cases"]) / len(self.results["test_cases"])
        self.results["summary"]["rag_similarity"] = rag_avg
        self.results["summary"]["llm_similarity"] = llm_avg
        self.results["summary"]["similarity_lift"] = rag_avg - llm_avg
        # Counts below threshold 0.2
        rag_below = sum(1 for tc in self.results["test_cases"] if tc.get("rag_similarity", 0.0) < 0.2)
        llm_below = sum(1 for tc in self.results["test_cases"] if tc.get("llm_similarity", 0.0) < 0.2)
        self.results["summary"]["rag_below_0_2"] = rag_below
        self.results["summary"]["llm_below_0_2"] = llm_below
    
    def _shorten(self, text: str, limit: int = 600) -> str:
        """Shorten long strings for terminal display."""
        try:
            s = (text or "").strip()
        except Exception:
            s = str(text)
        if len(s) <= limit:
            return s
        return s[:limit] + "…"
    
    def print_summary(self):
        """Print a summary of the test results"""
        summary = self.results["summary"]
        print("\n" + "="*50)
        print("RAG vs LLM Testing Results")
        print("="*50)
        print(f"Total questions: {summary['total_questions']}")
        print(f"RAG similarity: {summary['rag_similarity']:.2f}")
        print(f"LLM similarity: {summary['llm_similarity']:.2f}")
        print(f"Similarity lift: {summary['similarity_lift']:.2f}")
        print(f"RAG < 0.2: {summary.get('rag_below_0_2', 0)}")
        print(f"LLM < 0.2: {summary.get('llm_below_0_2', 0)}")
        print("="*50)
        
        # Print detailed results for each question
        print("\nDetailed Results:")
        for i, test_case in enumerate(self.results["test_cases"]):
            print(f"\nQuestion {i+1}: {test_case['question']}")
            print(f"  RAG similarity: {test_case['rag_similarity']:.2f}")
            print(f"  LLM similarity: {test_case['llm_similarity']:.2f}")
            print(f"  Lift: {test_case['similarity_lift']:.2f}")

async def main():
    """Main function to run the RAG tester"""
    parser = argparse.ArgumentParser(description="Test RAG vs non-retrieval LLM performance")
    parser.add_argument("--index", type=str, help="Name of the Pinecone index to use")
    parser.add_argument("--user", type=str, default="tester", help="User ID for the KBScraper")
    parser.add_argument("--output", type=str, help="Path to save test results")
    parser.add_argument("--test-file", type=str, help="Path to JSON file with test cases")
    parser.add_argument("--seed-context", action="store_true", help="Seed per-test context passages into the index before querying")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file if present")
    
    args = parser.parse_args()
    
    # Print which API keys are loaded (masked)
    def _mask(val: Optional[str]) -> str:
        if not val:
            return "<not set>"
        if len(val) <= 8:
            return val[0:2] + "***" + val[-2:]
        return val[0:4] + "..." + val[-4:]

    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    print(f"[Env] OPENAI_API_KEY = {_mask(openai_key)}")
    if pinecone_key:
        print(f"[Env] PINECONE_API_KEY = {_mask(pinecone_key)}")
    else:
        print("[Env] PINECONE_API_KEY = <not set>")
    
    # Load custom test cases if provided
    test_cases = DEFAULT_TEST_CASES
    if args.test_file:
        try:
            with open(args.test_file, 'r') as f:
                test_cases = json.load(f)
            logger.info(f"Loaded {len(test_cases)} test cases from {args.test_file}")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            sys.exit(1)
    
    # Create and run tester
    tester = RAGTester(
        index_name=args.index,
        user_id=args.user,
        test_cases=test_cases,
        output_file=args.output,
        seed_context=args.seed_context,
        resume=args.resume,
    )
    
    await tester.run_tests()
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
